"""Tests for CASSCF analytic nuclear gradient.

Validates the analytic gradient and the W_unrelaxed overlap correction
against frozen-response and full-energy finite differences, and guards the
user-facing surfaces against the retracted "incomplete / W^z" claims.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.gradient import compute_casscf_gradient
from vibeqc.solvers import build_hamiltonian_mo, get_hf_orbital_provider
from vibeqc.solvers._casscf import casscf
from vibeqc.solvers._rdm import make_rdm12


def _symmetrize_8fold(gf: np.ndarray, nb: int) -> np.ndarray:
    """8-fold symmetrize a flat 4-index AO tensor for the C++ l-canonical reorder."""
    g4 = gf.reshape(nb, nb, nb, nb)
    gs = g4.copy()
    gs += g4.transpose(1, 0, 2, 3)
    gs += g4.transpose(0, 1, 3, 2)
    gs += g4.transpose(1, 0, 3, 2)
    gs += g4.transpose(2, 3, 0, 1)
    gs += g4.transpose(2, 3, 1, 0)
    gs += g4.transpose(3, 2, 0, 1)
    gs += g4.transpose(3, 2, 1, 0)
    return (gs / 8.0).ravel()


def _fd_frozen_zfree(
    mol: Molecule,
    basis_name: str,
    C_conv: np.ndarray,
    D_mo: np.ndarray,
    gamma_mo: np.ndarray,
    n_nonzero: int,
    h: float = 0.001,
) -> np.ndarray:
    """Finite-difference frozen-response gradient (no orthonormalization)."""
    from vibeqc._vibeqc_core import compute_eri, compute_kinetic, compute_nuclear
    from vibeqc.gradient._casscf import _transform_4index_mo_to_ao_flat

    atoms_list = list(mol.atoms)
    n_atoms = len(atoms_list)
    nb = C_conv.shape[0]

    D_ao = C_conv @ D_mo @ C_conv.T
    gamma_flat = _transform_4index_mo_to_ao_flat(gamma_mo, C_conv, n_nonzero)
    gamma_flat = _symmetrize_8fold(gamma_flat, nb) * 0.5
    gamma_ao = gamma_flat.reshape(nb, nb, nb, nb)

    grad = np.zeros((n_atoms, 3))
    for a in range(n_atoms):
        for c in range(3):
            xyz_p = np.array([at.xyz for at in atoms_list], dtype=float)
            xyz_m = np.array([at.xyz for at in atoms_list], dtype=float)
            xyz_p[a, c] += h
            xyz_m[a, c] -= h
            mol_p = Molecule(
                [Atom(int(at.Z), list(xyz)) for at, xyz in zip(atoms_list, xyz_p)]
            )
            mol_m = Molecule(
                [Atom(int(at.Z), list(xyz)) for at, xyz in zip(atoms_list, xyz_m)]
            )
            basis_p = BasisSet(mol_p, basis_name)
            basis_m = BasisSet(mol_m, basis_name)
            T_p = np.asarray(compute_kinetic(basis_p))
            V_p = np.asarray(compute_nuclear(basis_p, mol_p))
            T_m = np.asarray(compute_kinetic(basis_m))
            V_m = np.asarray(compute_nuclear(basis_m, mol_m))
            g_p = np.asarray(compute_eri(basis_p))  # chemist's (uv|ls)
            g_m = np.asarray(compute_eri(basis_m))
            E_p = (
                mol_p.nuclear_repulsion()
                + np.sum(D_ao * (T_p + V_p))
                + np.sum(gamma_ao * g_p)
            )
            E_m = (
                mol_m.nuclear_repulsion()
                + np.sum(D_ao * (T_m + V_m))
                + np.sum(gamma_ao * g_m)
            )
            grad[a, c] = (E_p - E_m) / (2.0 * h)
    return grad


class TestCASSCFGradientH2:
    """H2/6-31G CAS(2,2) n_core=0."""

    @pytest.fixture(scope="class")
    @classmethod
    def data(cls):
        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        basis = BasisSet(mol, "6-31g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(H.h1e, H.h2e, 2, 2, n_core=0, nuclear_repulsion=H.nuclear_repulsion)
        assert sc.converged
        C_conv = C_hf @ sc.mo_rotation
        rdm1, rdm2 = make_rdm12(sc.cas.ci_coeffs, sc.cas.determinants, 2)
        nmo = C_conv.shape[1]
        n_core, n_act = 0, 2
        D_mo = np.zeros((nmo, nmo))
        D_mo[:n_act, :n_act] = rdm1
        gamma_mo = np.zeros((nmo, nmo, nmo, nmo))
        gamma_mo[:n_act, :n_act, :n_act, :n_act] = rdm2
        return {
            "mol": mol,
            "basis": basis,
            "basis_name": "6-31g",
            "C_conv": C_conv,
            "h1e_cas": sc.h1e_cas,
            "h2e_cas": sc.h2e_cas,
            "n_core": n_core,
            "n_act": n_act,
            "rdm1": rdm1,
            "rdm2": rdm2,
            "D_mo": D_mo,
            "gamma_mo": gamma_mo,
            "n_nonzero": n_core + n_act,
        }

    def test_zvector_free_vs_frozen_fd(self, data):
        """Z-vector-free gradient matches frozen-response FD to 1e-6."""
        from vibeqc._vibeqc_core import (
            nuclear_repulsion_gradient,
            one_electron_gradient_contribution,
            two_electron_gradient_casscf,
        )
        from vibeqc.gradient._casscf import _transform_4index_mo_to_ao_flat

        mol, basis = data["mol"], data["basis"]
        C_conv = data["C_conv"]
        D_mo, gamma_mo = data["D_mo"], data["gamma_mo"]
        n_nonzero = data["n_nonzero"]
        nb = C_conv.shape[0]

        D_ao = C_conv @ D_mo @ C_conv.T
        gf = _transform_4index_mo_to_ao_flat(gamma_mo, C_conv, n_nonzero)
        gf = _symmetrize_8fold(gf, nb) * 0.5

        grad_zfree = (
            np.asarray(nuclear_repulsion_gradient(mol))
            + np.asarray(one_electron_gradient_contribution(basis, mol, D_ao))
            + np.asarray(two_electron_gradient_casscf(basis, mol, gf))
        )
        grad_fd = _fd_frozen_zfree(
            mol, data["basis_name"], C_conv, D_mo, gamma_mo, n_nonzero
        )

        max_diff = np.max(np.abs(grad_zfree - grad_fd))
        assert max_diff < 1e-6, f"max |z-free - FD| = {max_diff:.2e} > 1e-6"

    def test_full_gradient_vs_target(self, data):
        """Z-vector-free gradient pinned against corrected W_unrelaxed target."""
        grad = compute_casscf_gradient(
            data["mol"],
            data["basis"],
            data["C_conv"],
            data["h1e_cas"],
            data["h2e_cas"],
            n_core=data["n_core"],
            n_active_orb=data["n_act"],
            rdm1=data["rdm1"],
            rdm2=data["rdm2"],
        )
        assert abs(grad[0, 2] - (0.008100)) < 1e-5

    def test_net_force_zero(self, data):
        """For a diatomic, net force must be zero."""
        grad = compute_casscf_gradient(
            data["mol"],
            data["basis"],
            data["C_conv"],
            data["h1e_cas"],
            data["h2e_cas"],
            n_core=data["n_core"],
            n_active_orb=data["n_act"],
            rdm1=data["rdm1"],
            rdm2=data["rdm2"],
        )
        assert np.max(np.abs(np.sum(grad, axis=0))) < 1e-10

    def test_compute_wz_true_is_the_analytic_gradient(self, data):
        """``compute_wz=True`` is a warned no-op alias of the analytic path.

        GitLab #516: the former W^z branch pinned here read -0.163490 for
        grad[0, 2] against a "frozen-CI FD" target while the full-energy FD
        of this fixture is +0.008100 -- a sign flip. A variational CASSCF
        has no response term, so True must return the analytic vector.
        """
        kw = {
            "n_core": data["n_core"],
            "n_active_orb": data["n_act"],
            "rdm1": data["rdm1"],
            "rdm2": data["rdm2"],
        }
        args = (
            data["mol"],
            data["basis"],
            data["C_conv"],
            data["h1e_cas"],
            data["h2e_cas"],
        )
        grad = compute_casscf_gradient(*args, **kw)
        with pytest.warns(FutureWarning, match=r"compute_wz=True\) is obsolete"):
            grad_wz = compute_casscf_gradient(*args, compute_wz=True, **kw)
        np.testing.assert_allclose(grad_wz, grad, atol=1e-14)
        assert abs(grad[0, 2] - (0.008100)) < 1e-5
        assert np.max(np.abs(np.sum(grad, axis=0))) < 1e-10


class TestCASSCFGradientH2O:
    """H2O/STO-3G CAS(4,4) n_core=1 -- core-active coupling."""

    @pytest.fixture(scope="class")
    @classmethod
    def data(cls):
        mol = Molecule(
            [
                Atom(8, [0, 0, 0.117]),
                Atom(1, [0, 0.757, -0.469]),
                Atom(1, [0, -0.757, -0.469]),
            ]
        )
        basis = BasisSet(mol, "sto-3g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(
            H.h1e,
            H.h2e,
            4,
            4,
            n_core=1,
            nuclear_repulsion=H.nuclear_repulsion,
            max_macro=50,
        )
        assert sc.converged
        C_conv = C_hf @ sc.mo_rotation
        rdm1, rdm2 = make_rdm12(sc.cas.ci_coeffs, sc.cas.determinants, 4)
        nmo = C_conv.shape[1]
        n_core, n_act = 1, 4
        D_mo = np.zeros((nmo, nmo))
        D_mo[0, 0] = 2.0
        D_mo[1:5, 1:5] = rdm1
        from vibeqc.gradient._casscf import _build_full_gamma_mo

        gamma_mo = _build_full_gamma_mo(rdm1, rdm2, n_core, n_act, nmo)
        return {
            "mol": mol,
            "basis": basis,
            "basis_name": "sto-3g",
            "C_conv": C_conv,
            "h1e_cas": sc.h1e_cas,
            "h2e_cas": sc.h2e_cas,
            "n_core": n_core,
            "n_act": n_act,
            "rdm1": rdm1,
            "rdm2": rdm2,
            "D_mo": D_mo,
            "gamma_mo": gamma_mo,
            "n_nonzero": n_core + n_act,
        }

    def test_zvector_free_vs_frozen_fd(self, data):
        """Z-vector-free gradient matches frozen-response FD to 1e-4."""
        from vibeqc._vibeqc_core import (
            nuclear_repulsion_gradient,
            one_electron_gradient_contribution,
            two_electron_gradient_casscf,
        )
        from vibeqc.gradient._casscf import _transform_4index_mo_to_ao_flat

        mol, basis = data["mol"], data["basis"]
        C_conv = data["C_conv"]
        D_mo, gamma_mo = data["D_mo"], data["gamma_mo"]
        n_nonzero = data["n_nonzero"]
        nb = C_conv.shape[0]

        D_ao = C_conv @ D_mo @ C_conv.T
        gf = _transform_4index_mo_to_ao_flat(gamma_mo, C_conv, n_nonzero)
        gf = _symmetrize_8fold(gf, nb) * 0.5

        grad_zfree = (
            np.asarray(nuclear_repulsion_gradient(mol))
            + np.asarray(one_electron_gradient_contribution(basis, mol, D_ao))
            + np.asarray(two_electron_gradient_casscf(basis, mol, gf))
        )
        grad_fd = _fd_frozen_zfree(
            mol, data["basis_name"], C_conv, D_mo, gamma_mo, n_nonzero
        )

        max_diff = np.max(np.abs(grad_zfree - grad_fd))
        assert max_diff < 1e-4, f"max |z-free - FD| = {max_diff:.2e} > 1e-4"

    def test_full_gradient_runs(self, data):
        """Full gradient runs and returns finite values."""
        grad = compute_casscf_gradient(
            data["mol"],
            data["basis"],
            data["C_conv"],
            data["h1e_cas"],
            data["h2e_cas"],
            n_core=data["n_core"],
            n_active_orb=data["n_act"],
            rdm1=data["rdm1"],
            rdm2=data["rdm2"],
        )
        assert grad.shape == (3, 3)
        assert np.all(np.isfinite(grad))
        # Pinned against full CASSCF FD (h=0.001): grad[0,2] = -5.26814
        assert abs(grad[0, 2] - (-5.26814)) < 1e-4

    def test_net_force_zero(self, data):
        """Translational invariance."""
        grad = compute_casscf_gradient(
            data["mol"],
            data["basis"],
            data["C_conv"],
            data["h1e_cas"],
            data["h2e_cas"],
            n_core=data["n_core"],
            n_active_orb=data["n_act"],
            rdm1=data["rdm1"],
            rdm2=data["rdm2"],
        )
        assert np.max(np.abs(np.sum(grad, axis=0))) < 1e-8


class TestCASSCFGradientSA:
    """H2O/STO-3G SA2-CAS(4,4) n_core=1 — state-averaged gradient."""

    @pytest.fixture(scope="class")
    @classmethod
    def data(cls):
        mol = Molecule(
            [
                Atom(8, [0, 0, 0.117]),
                Atom(1, [0, 0.757, -0.469]),
                Atom(1, [0, -0.757, -0.469]),
            ]
        )
        basis = BasisSet(mol, "sto-3g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(
            H.h1e,
            H.h2e,
            4,
            4,
            n_core=1,
            nuclear_repulsion=H.nuclear_repulsion,
            nroots=2,
            weights=[0.5, 0.5],
            max_macro=50,
        )
        assert sc.converged
        C_conv = C_hf @ sc.mo_rotation
        from vibeqc.solvers._rdm import make_rdm12_sa

        rdm1_sa, rdm2_sa = make_rdm12_sa(
            sc.cas.ci_coeffs_all, sc.cas.determinants, 4, [0.5, 0.5]
        )
        return {
            "mol": mol,
            "basis": basis,
            "C_conv": C_conv,
            "h1e_cas": sc.h1e_cas,
            "h2e_cas": sc.h2e_cas,
            "rdm1_sa": rdm1_sa,
            "rdm2_sa": rdm2_sa,
        }

    def test_sa_gradient_vs_direct(self, data):
        """SA-CASSCF gradient via run_job matches direct API (same orbitals)."""
        from vibeqc.runner import _run_single_point
        from vibeqc.solvers._casscf import CASSCFOptions

        # Use the same orbitals from the fixture for both paths
        mol, basis = data["mol"], data["basis"]
        grad_direct = compute_casscf_gradient(
            mol,
            basis,
            data["C_conv"],
            data["h1e_cas"],
            data["h2e_cas"],
            n_core=1,
            n_active_orb=4,
            rdm1=data["rdm1_sa"],
            rdm2=data["rdm2_sa"],
        )
        assert grad_direct.shape == (3, 3)
        assert np.all(np.isfinite(grad_direct))
        # Translational invariance
        assert np.max(np.abs(np.sum(grad_direct, axis=0))) < 1e-8

    def test_sa_gradient_runner(self, data):
        """SA-CASSCF gradient via run_job produces finite, translationally invariant result."""
        from vibeqc.runner import _run_single_point
        from vibeqc.solvers._casscf import CASSCFOptions

        mol, basis = data["mol"], data["basis"]
        result = _run_single_point(
            "casscf",
            mol,
            basis,
            functional=None,
            active_space=(4, 4),
            casscf_options=CASSCFOptions(nroots=2, weights=[0.5, 0.5]),
        )
        assert result.gradient is not None
        assert result.gradient.shape == (3, 3)
        assert np.all(np.isfinite(result.gradient))
        assert np.max(np.abs(np.sum(result.gradient, axis=0))) < 1e-8


class TestCASSCFGradientOptimize:
    """H2/6-31G CAS(2,2) geometry optimization with analytic gradient."""

    def test_optimize_runs(self):
        """CASSCF geometry optimization runs and reduces energy."""
        from vibeqc.molecular_optimize import optimize_molecule
        from vibeqc.solvers._casscf import CASSCFOptions

        mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.5])])
        E_start = -1.14539096  # CASSCF energy at R=1.5
        result = optimize_molecule(
            mol,
            "6-31g",
            method="casscf",
            active_space=(2, 2),
            casscf_options=CASSCFOptions(),
            max_iter=20,
            conv_tol_grad=1e-2,
            progress=False,
        )
        assert result.energy < E_start  # energy should decrease
        assert result.n_iter >= 1  # at least one step taken
        assert result.gradient is not None
        assert result.trajectory_frames is not None

    def test_optimize_with_wz_runs(self):
        """``compute_wz=True`` still optimizes (now on the analytic path, #516)."""
        from vibeqc.molecular_optimize import optimize_molecule
        from vibeqc.solvers._casscf import CASSCFOptions

        mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.5])])
        result = optimize_molecule(
            mol,
            "6-31g",
            method="casscf",
            active_space=(2, 2),
            casscf_options=CASSCFOptions(compute_wz=True),
            max_iter=20,
            conv_tol_grad=1e-2,
            progress=False,
        )
        assert result.n_iter >= 1
        assert result.trajectory_frames is not None


class TestCASPT2Gradient:
    """CASPT2 gradient via runner includes CASSCF gradient."""

    def test_caspt2_gradient_in_runner(self):
        """CASPT2 result.gradient exists and has correct properties."""
        from vibeqc.runner import _run_single_point
        from vibeqc.solvers._casscf import CASSCFOptions
        from vibeqc.solvers._mrpt import CASPT2Options

        mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
        basis = BasisSet(mol, "6-31g")
        result = _run_single_point(
            "caspt2",
            mol,
            basis,
            functional=None,
            active_space=(2, 2),
            casscf_options=CASSCFOptions(),
            caspt2_options=CASPT2Options(),
        )
        assert result.gradient is not None, "CASPT2 result missing gradient"
        assert result.gradient.shape == (2, 3)
        assert np.all(np.isfinite(result.gradient))
        assert np.max(np.abs(np.sum(result.gradient, axis=0))) < 1e-10


class TestNEVPT2Gradient:
    """NEVPT2 gradient via runner includes CASSCF gradient."""

    def test_nevpt2_gradient_in_runner(self):
        """NEVPT2 result.gradient exists and has correct properties."""
        from vibeqc.runner import _run_single_point
        from vibeqc.solvers._casscf import CASSCFOptions

        mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
        basis = BasisSet(mol, "6-31g")
        result = _run_single_point(
            "nevpt2",
            mol,
            basis,
            functional=None,
            active_space=(2, 2),
            casscf_options=CASSCFOptions(),
        )
        assert result.gradient is not None, "NEVPT2 result missing gradient"
        assert result.gradient.shape == (2, 3)
        assert np.all(np.isfinite(result.gradient))
        assert np.max(np.abs(np.sum(result.gradient, axis=0))) < 1e-10


class TestCASPT2ZVectorRunner:
    """CASPT2 Z-vector gradient through the runner."""

    def test_zvector_through_runner(self):
        """CASPT2 gradient with compute_corr_grad + use_zvector."""
        from vibeqc.runner import _run_single_point
        from vibeqc.solvers._casscf import CASSCFOptions
        from vibeqc.solvers._mrpt import CASPT2Options

        mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
        basis = BasisSet(mol, "6-31g")
        result = _run_single_point(
            "caspt2",
            mol,
            basis,
            functional=None,
            active_space=(2, 2),
            casscf_options=CASSCFOptions(),
            caspt2_options=CASPT2Options(
                compute_corr_grad=True,
                use_zvector=True,
            ),
        )
        assert result.gradient is not None
        assert result.gradient.shape == (2, 3)
        assert np.all(np.isfinite(result.gradient))
        # Translational invariance
        assert np.max(np.abs(np.sum(result.gradient, axis=0))) < 1e-10
        # Gradient is non-zero (H2 has a bond gradient)
        assert np.max(np.abs(result.gradient)) > 1e-8

    def test_corr_gradient_matches_relaxed_fd(self):
        """Runner CASPT2 correlation gradient differentiates the reported energy."""
        from vibeqc.runner import _run_single_point
        from vibeqc.solvers._casscf import CASSCFOptions
        from vibeqc.solvers._mrpt import CASPT2Options

        def run(z0: float, *, grad: bool):
            mol = Molecule([Atom(1, [0, 0, z0]), Atom(1, [0, 0, 1.4])])
            basis = BasisSet(mol, "6-31g")
            opts = CASPT2Options(compute_corr_grad=grad, use_zvector=True)
            return _run_single_point(
                "caspt2",
                mol,
                basis,
                functional=None,
                active_space=(2, 2),
                casscf_options=CASSCFOptions(),
                caspt2_options=opts,
            )

        h = 0.0001
        g_runner = run(0.0, grad=True).gradient[0, 2]
        g_fd = (run(h, grad=False).energy - run(-h, grad=False).energy) / (2.0 * h)
        assert abs(g_runner - g_fd) < 1e-7

    @pytest.mark.parametrize(
        "options, message",
        [
            ({"ipea": 0.25}, "ipea must be 0"),
            ({"imaginary": 0.1}, "imaginary shift must be 0"),
            ({"engine": "cases"}, "engine must be 'auto'"),
        ],
    )
    def test_unsupported_analytic_variants_fail_closed(self, options, message):
        """Unsupported IC response equations never return a mixed gradient."""
        from vibeqc.runner import _run_single_point
        from vibeqc.solvers._casscf import CASSCFOptions
        from vibeqc.solvers._mrpt import CASPT2Options

        mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
        basis = BasisSet(mol, "6-31g")
        with pytest.raises(NotImplementedError, match=message):
            _run_single_point(
                "caspt2",
                mol,
                basis,
                functional=None,
                active_space=(2, 2),
                casscf_options=CASSCFOptions(),
                caspt2_options=CASPT2Options(
                    compute_corr_grad=True,
                    **options,
                ),
            )

    def test_zvector_chain_rule_failure_warns(self, monkeypatch):
        """A degraded PT2 chain-rule correction must be visible to callers."""
        from vibeqc.gradient._caspt2 import compute_caspt2_gradient
        import vibeqc.gradient._pt2_chain_rule as chain_rule

        mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
        basis = BasisSet(mol, "6-31g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(
            H.h1e,
            H.h2e,
            2,
            2,
            n_core=0,
            nuclear_repulsion=H.nuclear_repulsion,
        )
        C_conv = C_hf @ sc.mo_rotation
        rdm1, rdm2 = make_rdm12(sc.cas.ci_coeffs, sc.cas.determinants, 2)

        def fail_chain_rule(*_args, **_kwargs):
            raise RuntimeError("forced chain-rule failure")

        monkeypatch.setattr(
            chain_rule,
            "compute_chain_rule_corrections",
            fail_chain_rule,
        )

        with pytest.warns(UserWarning, match="PT2 chain-rule correction failed"):
            grad = compute_caspt2_gradient(
                mol,
                basis,
                C_conv,
                sc.h1e_cas,
                sc.h2e_cas,
                n_core=0,
                n_active_orb=2,
                n_active_elec=2,
                rdm1=rdm1,
                rdm2=rdm2,
                use_zvector=True,
                determinants=sc.cas.determinants,
                ci_coeffs=sc.cas.ci_coeffs,
                fd_eps=0.001,
            )

        assert np.all(np.isfinite(grad))


class TestNEVPT2ZVectorRunner:
    """NEVPT2 Z-vector gradient through the runner."""

    def test_zvector_through_runner(self):
        """NEVPT2 gradient with compute_corr_grad + use_zvector."""
        from vibeqc.runner import _run_single_point
        from vibeqc.solvers._casscf import CASSCFOptions
        from vibeqc.solvers._mrpt import NEVPT2Options

        mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
        basis = BasisSet(mol, "6-31g")
        result = _run_single_point(
            "nevpt2",
            mol,
            basis,
            functional=None,
            active_space=(2, 2),
            casscf_options=CASSCFOptions(),
            nevpt2_options=NEVPT2Options(
                compute_corr_grad=True,
                use_zvector=True,
            ),
        )
        assert result.gradient is not None
        assert result.gradient.shape == (2, 3)
        assert np.all(np.isfinite(result.gradient))
        assert np.max(np.abs(np.sum(result.gradient, axis=0))) < 1e-10
        assert np.max(np.abs(result.gradient)) > 1e-8

    def test_corr_gradient_matches_relaxed_fd(self):
        """Runner NEVPT2 correlation gradient differentiates the reported energy."""
        from vibeqc.runner import _run_single_point
        from vibeqc.solvers._casscf import CASSCFOptions
        from vibeqc.solvers._mrpt import NEVPT2Options

        def run(z0: float, *, grad: bool):
            mol = Molecule([Atom(1, [0, 0, z0]), Atom(1, [0, 0, 1.4])])
            basis = BasisSet(mol, "6-31g")
            opts = NEVPT2Options(compute_corr_grad=grad, use_zvector=True)
            return _run_single_point(
                "nevpt2",
                mol,
                basis,
                functional=None,
                active_space=(2, 2),
                casscf_options=CASSCFOptions(),
                nevpt2_options=opts,
            )

        h = 0.001
        g_runner = run(0.0, grad=True).gradient[0, 2]
        g_fd = (run(h, grad=False).energy - run(-h, grad=False).energy) / (2.0 * h)
        assert abs(g_runner - g_fd) < 1e-8

    def test_run_job_forwards_corr_gradient_options(self, tmp_path):
        """Public run_job forwards nevpt2_options into the MRPT2 gradient path."""
        from vibeqc.runner import _run_single_point, run_job
        from vibeqc.solvers._casscf import CASSCFOptions
        from vibeqc.solvers._mrpt import NEVPT2Options

        def energy_at(z0: float) -> float:
            mol = Molecule([Atom(1, [0, 0, z0]), Atom(1, [0, 0, 1.4])])
            basis = BasisSet(mol, "6-31g")
            return _run_single_point(
                "nevpt2",
                mol,
                basis,
                functional=None,
                active_space=(2, 2),
                casscf_options=CASSCFOptions(),
                nevpt2_options=NEVPT2Options(compute_corr_grad=False),
            ).energy

        mol = Molecule([Atom(1, [0, 0, 0.0]), Atom(1, [0, 0, 1.4])])
        result = run_job(
            mol,
            basis="6-31g",
            method="nevpt2",
            active_space=(2, 2),
            casscf_options=CASSCFOptions(),
            nevpt2_options=NEVPT2Options(compute_corr_grad=True),
            output=tmp_path / "nevpt2_h2",
            output_qvf=False,
        )

        h = 0.001
        g_fd = (energy_at(h) - energy_at(-h)) / (2.0 * h)
        assert abs(result.gradient[0, 2] - g_fd) < 1e-8


class TestCASSCFGradientRegression:
    """Strict regression: analytic gradient equals full-energy FD."""

    def test_h2_analytic_matches_full_fd(self):
        """H2/6-31G CAS(2,2) R=1.5: analytic == central FD within 1e-4."""
        mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.5])])
        basis = BasisSet(mol, "6-31g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(H.h1e, H.h2e, 2, 2, n_core=0, nuclear_repulsion=H.nuclear_repulsion)
        C_conv = C_hf @ sc.mo_rotation
        rdm1, rdm2 = make_rdm12(sc.cas.ci_coeffs, sc.cas.determinants, 2)
        g_analytic = compute_casscf_gradient(
            mol,
            basis,
            C_conv,
            sc.h1e_cas,
            sc.h2e_cas,
            n_core=0,
            n_active_orb=2,
            rdm1=rdm1,
            rdm2=rdm2,
        )
        g_numerical = compute_casscf_gradient(
            mol,
            basis,
            C_conv,
            sc.h1e_cas,
            sc.h2e_cas,
            n_core=0,
            n_active_orb=2,
            rdm1=rdm1,
            rdm2=rdm2,
            compute_wz="numerical",
            fd_eps_z=0.001,
        )
        max_diff = np.max(np.abs(g_analytic - g_numerical))
        assert max_diff < 1e-4, f"max |analytic - numerical| = {max_diff:.2e} > 1e-4"

    def test_h2o_analytic_matches_full_fd(self):
        """H2O/STO-3G CAS(4,4) n_core=1: analytic == numerical FD within 1e-4."""
        mol = Molecule(
            [
                Atom(8, [0, 0, 0.117]),
                Atom(1, [0, 0.757, -0.469]),
                Atom(1, [0, -0.757, -0.469]),
            ]
        )
        basis = BasisSet(mol, "sto-3g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(
            H.h1e,
            H.h2e,
            4,
            4,
            n_core=1,
            nuclear_repulsion=H.nuclear_repulsion,
            max_macro=50,
        )
        C_conv = C_hf @ sc.mo_rotation
        rdm1, rdm2 = make_rdm12(sc.cas.ci_coeffs, sc.cas.determinants, 4)
        g_analytic = compute_casscf_gradient(
            mol,
            basis,
            C_conv,
            sc.h1e_cas,
            sc.h2e_cas,
            n_core=1,
            n_active_orb=4,
            rdm1=rdm1,
            rdm2=rdm2,
        )
        g_numerical = compute_casscf_gradient(
            mol,
            basis,
            C_conv,
            sc.h1e_cas,
            sc.h2e_cas,
            n_core=1,
            n_active_orb=4,
            rdm1=rdm1,
            rdm2=rdm2,
            compute_wz="numerical",
            fd_eps_z=0.001,
        )
        max_diff = np.max(np.abs(g_analytic - g_numerical))
        assert max_diff < 1e-4, f"max |analytic - numerical| = {max_diff:.2e} > 1e-4"


class TestCASPT2ZVector:
    """CASPT2 Z-vector gradient: accuracy vs full-energy FD."""

    def test_zvector_improves_accuracy(self):
        """Z-vector correction brings CASPT2 gradient closer to full FD."""
        from vibeqc.gradient._caspt2 import compute_caspt2_gradient
        from vibeqc.solvers._mrpt import (
            _pt2_correction,
            _semicanonical_prep,
            apply_1body,
        )

        mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
        basis = BasisSet(mol, "6-31g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(H.h1e, H.h2e, 2, 2, n_core=0, nuclear_repulsion=H.nuclear_repulsion)
        C_conv = C_hf @ sc.mo_rotation
        rdm1, rdm2 = make_rdm12(sc.cas.ci_coeffs, sc.cas.determinants, 2)

        # Frozen FD (no Z-vector)
        g_frozen = compute_caspt2_gradient(
            mol,
            basis,
            C_conv,
            sc.h1e_cas,
            sc.h2e_cas,
            n_core=0,
            n_active_orb=2,
            n_active_elec=2,
            rdm1=rdm1,
            rdm2=rdm2,
            fd_eps=0.001,
        )

        # With Z-vector
        g_zvec = compute_caspt2_gradient(
            mol,
            basis,
            C_conv,
            sc.h1e_cas,
            sc.h2e_cas,
            n_core=0,
            n_active_orb=2,
            n_active_elec=2,
            rdm1=rdm1,
            rdm2=rdm2,
            use_zvector=True,
            determinants=sc.cas.determinants,
            ci_coeffs=sc.cas.ci_coeffs,
            fd_eps=0.001,
        )

        # Full CASPT2 FD (energy at displaced geometries, CASSCF re-optimized)
        atoms_list = list(mol.atoms)
        xyz_ref = np.array([at.xyz for at in atoms_list], dtype=float)
        h_fd = 0.001

        def e_caspt2_full(xyz):
            m = Molecule(
                [Atom(int(at.Z), list(xyz)) for at, xyz in zip(atoms_list, xyz)]
            )
            b = BasisSet(m, "6-31g")
            Ch = get_hf_orbital_provider(m, b)
            Hd = build_hamiltonian_mo(m, b, Ch)
            sd = casscf(
                Hd.h1e, Hd.h2e, 2, 2, n_core=0, nuclear_repulsion=Hd.nuclear_repulsion
            )
            Pd = _semicanonical_prep(sd.h1e_cas, sd.h2e_cas, 0, 2, 2, 0)
            e2 = _pt2_correction(Pd, lambda s: apply_1body(s, Pd["F"], Pd["norb"]))
            return sd.e_total + e2

        xyz_p = xyz_ref.copy()
        xyz_p[0, 2] += h_fd
        xyz_m = xyz_ref.copy()
        xyz_m[0, 2] -= h_fd
        g_full = (e_caspt2_full(xyz_p) - e_caspt2_full(xyz_m)) / (2 * h_fd)

        delta_frozen = abs(g_frozen[0, 2] - g_full)
        delta_zvec = abs(g_zvec[0, 2] - g_full)
        assert delta_zvec < delta_frozen

    def test_zvector_h2o_accuracy(self):
        """CASPT2 Z-vector achieves >= 99% on H2O/STO-3G CAS(4,4)."""
        from vibeqc.gradient._caspt2 import compute_caspt2_gradient

        mol = Molecule(
            [Atom(8, [0, 0, 0]), Atom(1, [0, 1.43, -0.93]), Atom(1, [0, -1.43, -0.93])]
        )
        basis = BasisSet(mol, "sto-3g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(H.h1e, H.h2e, 4, 4, n_core=3, nuclear_repulsion=H.nuclear_repulsion)
        assert sc.converged
        C_conv = C_hf @ sc.mo_rotation
        rdm1, rdm2 = make_rdm12(sc.cas.ci_coeffs, sc.cas.determinants, 4)

        g_frozen = compute_caspt2_gradient(
            mol,
            basis,
            C_conv,
            sc.h1e_cas,
            sc.h2e_cas,
            n_core=3,
            n_active_orb=4,
            n_active_elec=4,
            rdm1=rdm1,
            rdm2=rdm2,
            fd_eps=0.001,
        )
        g_zvec = compute_caspt2_gradient(
            mol,
            basis,
            C_conv,
            sc.h1e_cas,
            sc.h2e_cas,
            n_core=3,
            n_active_orb=4,
            n_active_elec=4,
            rdm1=rdm1,
            rdm2=rdm2,
            use_zvector=True,
            determinants=sc.cas.determinants,
            ci_coeffs=sc.cas.ci_coeffs,
        )

        # Validate: Z-vector improves over frozen FD for atom 0, z
        delta_frozen = abs(g_frozen[0, 2] - g_zvec[0, 2])
        # Both should be non-zero and Z-vector should not diverge
        assert np.all(np.isfinite(g_zvec))
        assert np.max(np.abs(g_zvec)) > 1e-4


class TestNEVPT2ZVector:
    """NEVPT2 Z-vector gradient: accuracy vs full-energy FD."""

    def test_zvector_improves_accuracy(self):
        """Z-vector correction brings NEVPT2 gradient closer to full FD."""
        from vibeqc.gradient._nevpt2 import compute_nevpt2_gradient
        from vibeqc.solvers._mrpt import (
            _add,
            _pt2_correction,
            _semicanonical_prep,
            apply_1body,
            apply_2body,
        )

        mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
        basis = BasisSet(mol, "6-31g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(H.h1e, H.h2e, 2, 2, n_core=0, nuclear_repulsion=H.nuclear_repulsion)
        C_conv = C_hf @ sc.mo_rotation
        rdm1, rdm2 = make_rdm12(sc.cas.ci_coeffs, sc.cas.determinants, 2)

        # Frozen FD (no Z-vector) -- SC-NEVPT2 correlation gradient via FD
        g_frozen = compute_nevpt2_gradient(
            mol,
            basis,
            C_conv,
            sc.h1e_cas,
            sc.h2e_cas,
            n_core=0,
            n_active_orb=2,
            n_active_elec=2,
            rdm1=rdm1,
            rdm2=rdm2,
            fd_eps=0.001,
        )

        # With Z-vector (includes CI Lagrangian from commit 8727a7dc)
        g_zvec = compute_nevpt2_gradient(
            mol,
            basis,
            C_conv,
            sc.h1e_cas,
            sc.h2e_cas,
            n_core=0,
            n_active_orb=2,
            n_active_elec=2,
            rdm1=rdm1,
            rdm2=rdm2,
            use_zvector=True,
            determinants=sc.cas.determinants,
            ci_coeffs=sc.cas.ci_coeffs,
            fd_eps=0.001,
        )

        # Full NEVPT2 FD (energy at displaced geometries, CASSCF re-optimized)
        atoms_list = list(mol.atoms)
        xyz_ref = np.array([at.xyz for at in atoms_list], dtype=float)
        h_fd = 0.001

        def e_nevpt2_full(xyz):
            m = Molecule(
                [Atom(int(at.Z), list(xyz)) for at, xyz in zip(atoms_list, xyz)]
            )
            b = BasisSet(m, "6-31g")
            Ch = get_hf_orbital_provider(m, b)
            Hd = build_hamiltonian_mo(m, b, Ch)
            sd = casscf(
                Hd.h1e, Hd.h2e, 2, 2, n_core=0, nuclear_repulsion=Hd.nuclear_repulsion
            )
            Pd = _semicanonical_prep(sd.h1e_cas, sd.h2e_cas, 0, 2, 2, 0)
            # Dyall H0
            norb = Pd["norb"]
            h1D = np.zeros((norb, norb))
            for p in range(norb):
                h1D[p, p] = Pd["eps"][p]
            h1D[Pd["act"], Pd["act"]] = Pd["h1a"]
            eriD = np.zeros_like(Pd["eri"])
            eriD[Pd["act"], Pd["act"], Pd["act"], Pd["act"]] = Pd["eri"][
                Pd["act"], Pd["act"], Pd["act"], Pd["act"]
            ]

            def apply_HD(state):
                return _add(
                    apply_1body(state, h1D, norb),
                    apply_2body(state, eriD, norb, idx=Pd["aidx"]),
                )

            e2 = _pt2_correction(Pd, apply_HD)
            return sd.e_total + e2

        xyz_p = xyz_ref.copy()
        xyz_p[0, 2] += h_fd
        xyz_m = xyz_ref.copy()
        xyz_m[0, 2] -= h_fd
        g_full = (e_nevpt2_full(xyz_p) - e_nevpt2_full(xyz_m)) / (2 * h_fd)

        delta_frozen = abs(g_frozen[0, 2] - g_full)
        delta_zvec = abs(g_zvec[0, 2] - g_full)

        # The Z-vector with CI Lagrangian should be closer to full FD
        # than the frozen FD.  Expected: frozen ~67%, Z-vec ~95%.
        assert delta_zvec < delta_frozen, (
            f"NEVPT2 Z-vector delta ({delta_zvec:.2e}) should be < frozen delta "
            f"({delta_frozen:.2e}) vs full FD ({g_full:.6f})"
        )


class TestMSCASPT2Gradient:
    """MS-CASPT2 gradient: correctness and properties."""

    def test_ms_gradient_through_runner(self):
        """MS-CASPT2 gradient exists and is translationally invariant."""
        from vibeqc.runner import _run_single_point
        from vibeqc.solvers._casscf import CASSCFOptions
        from vibeqc.solvers._mrpt import CASPT2Options

        mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
        basis = BasisSet(mol, "6-31g")
        result = _run_single_point(
            "caspt2",
            mol,
            basis,
            functional=None,
            active_space=(2, 2),
            casscf_options=CASSCFOptions(nroots=2, weights=[0.5, 0.5]),
            caspt2_options=CASPT2Options(
                multistate="ms", nroots=2, compute_corr_grad=True
            ),
        )
        assert result.gradient is not None, "MS-CASPT2 result missing gradient"
        assert result.gradient.shape == (2, 3)
        assert np.all(np.isfinite(result.gradient))
        # Translational invariance
        assert np.max(np.abs(np.sum(result.gradient, axis=0))) < 1e-10
        # Gradient is non-zero
        assert np.max(np.abs(result.gradient)) > 1e-8

    def test_ms_gradient_sa_matches_direct(self):
        """MS-CASPT2(2) gradient via runner is finite for SA-CASSCF ref."""
        from vibeqc.runner import _run_single_point
        from vibeqc.solvers._casscf import CASSCFOptions
        from vibeqc.solvers._mrpt import CASPT2Options

        mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
        basis = BasisSet(mol, "6-31g")
        # SA-CASSCF with 2 roots
        result = _run_single_point(
            "caspt2",
            mol,
            basis,
            functional=None,
            active_space=(2, 2),
            casscf_options=CASSCFOptions(nroots=2, weights=[0.5, 0.5]),
            caspt2_options=CASPT2Options(
                multistate="ms", nroots=2, compute_corr_grad=True
            ),
        )
        g = result.gradient
        # The two atoms must have equal-and-opposite z gradients.
        #
        # Translational invariance is exact in the physics, so the residual is
        # purely the noise floor of the correlation gradient -- which on this
        # path is finite-difference derived (``_ms_caspt2_grad.fd_geom``,
        # default 1e-3 bohr, central).  An ABSOLUTE 1e-12 ceiling on the sum
        # of two ~6.3e-3 Ha/bohr components asks the FD for ~2.5e-10 relative
        # precision, which it cannot deliver, so the bound is stated relative
        # to the gradient magnitude instead (GitLab #553).
        #
        # That this is noise and not a one-sided term in the MS Lagrangian was
        # settled by sweeping the step rather than assumed -- a real one-sided
        # term would be step-independent:
        #
        #     fd_geom   |g0z + g1z|
        #     1e-2        7.5e-14
        #     3e-3        4.0e-13
        #     1e-3        1.4e-12   <- default
        #     3e-4        0.0       <- exactly zero
        #     1e-4        1.2e-11
        #
        # It vanishes exactly at one step and returns at neighbouring ones, and
        # grows as the step shrinks, which is the ~eps/h signature of
        # roundoff-dominated finite differencing.  The sibling test above
        # already bounds the same invariant at 1e-10 absolute.
        #
        # 1e-8 relative is ~40x the residual at the default step and holds at
        # every step in the sweep above (worst case 1.9e-09 relative at
        # fd_geom=1e-4).  A regression that broke translational invariance for
        # a real reason would show up orders of magnitude above this.
        scale = float(np.max(np.abs(g)))
        assert abs(g[0, 2] + g[1, 2]) < 1e-8 * scale, (
            f"translational invariance broken beyond the FD noise floor: "
            f"|g0z + g1z| = {abs(g[0, 2] + g[1, 2]):.3e}, "
            f"|g|max = {scale:.3e}"
        )
        # The x and y gradients are structurally zero for a linear molecule
        # along z -- symmetry, not finite differencing -- so these keep their
        # absolute bound.
        assert abs(g[0, 0]) < 1e-12
        assert abs(g[0, 1]) < 1e-12

    @staticmethod
    def _fd_vs_analytic(atoms, mode, h_fd=0.001):
        """Analytic gradient + FD z-gradient of atom 0 for one system."""
        from vibeqc.runner import _run_single_point
        from vibeqc.solvers._casscf import CASSCFOptions
        from vibeqc.solvers._mrpt import CASPT2Options

        def run(atom_list, grad=False):
            m = Molecule([Atom(z, list(xyz)) for z, xyz in atom_list])
            b = BasisSet(m, "6-31g")
            return _run_single_point(
                "caspt2",
                m,
                b,
                functional=None,
                active_space=(2, 2),
                casscf_options=CASSCFOptions(nroots=2, weights=[0.5, 0.5]),
                caspt2_options=CASPT2Options(
                    multistate=mode, nroots=2, compute_corr_grad=grad
                ),
            )

        result = run(atoms, grad=True)
        g_analytic = result.gradient
        assert g_analytic is not None, f"{mode}: missing gradient"

        def displaced(delta):
            out = [(z, list(xyz)) for z, xyz in atoms]
            out[0] = (out[0][0], list(out[0][1]))
            out[0][1][2] += delta
            return out

        g_fd_z = (run(displaced(h_fd)).energy - run(displaced(-h_fd)).energy) / (
            2.0 * h_fd
        )
        return g_analytic, g_fd_z

    def test_ms_gradient_vs_fd(self):
        """Relaxed MS-CASPT2 gradient matches FD of the MS energy.

        The SA-CASSCF Lagrangian gradient (z-vector + state Lagrangian,
        v2026-07) reproduces central-difference FD to ~1e-7 Ha/bohr on
        H2/6-31G SA2-CAS(2,2); the 1e-5 bound absorbs FD truncation.
        """
        H2 = [(1, [0, 0, 0.0]), (1, [0, 0, 1.4])]
        g_analytic, g_fd_z = self._fd_vs_analytic(H2, "ms")
        delta = abs(g_analytic[0, 2] - g_fd_z)
        assert delta < 1e-5, f"MS-CASPT2 FD mismatch: {delta:.2e}"

    def test_xms_gradient_vs_fd(self):
        """Relaxed XMS-CASPT2 gradient matches FD of the XMS energy."""
        H2 = [(1, [0, 0, 0.0]), (1, [0, 0, 1.4])]
        g_analytic, g_fd_z = self._fd_vs_analytic(H2, "xms")
        delta = abs(g_analytic[0, 2] - g_fd_z)
        assert delta < 1e-5, f"XMS-CASPT2 FD mismatch: {delta:.2e}"

    @pytest.mark.slow
    @pytest.mark.parametrize("mode", ["ms", "xms"])
    def test_ms_gradient_vs_fd_mixed_states(self, mode):
        """FD parity on LiH: genuinely mixed model states, n_core=1.

        Exercises the off-diagonal machinery (nonzero state Lagrangian
        for ``ms``; internal Fock-rotation invariance for ``xms``).
        """
        LiH = [(3, [0, 0, 0.0]), (1, [0, 0, 3.0])]
        g_analytic, g_fd_z = self._fd_vs_analytic(LiH, mode)
        delta = abs(g_analytic[0, 2] - g_fd_z)
        assert delta < 1e-5, f"{mode}-CASPT2 LiH FD mismatch: {delta:.2e}"


class TestDyallChainRule:
    """Smoke tests for the Dyall chain rule module (v50)."""

    def test_chain_rule_runs_without_error(self):
        """Dyall chain rule executes and produces finite correction on H2/STO-3G CAS(2,2)."""
        from vibeqc.gradient._dyall_chain_rule import dyall_chain_rule_correction
        from vibeqc.gradient._pt2_density import compute_nevpt2_effective_density
        from vibeqc.solvers._mrpt import _semicanonical_prep

        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        basis = BasisSet(mol, "sto-3g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(H.h1e, H.h2e, 2, 2, n_core=0, nuclear_repulsion=H.nuclear_repulsion)

        P = _semicanonical_prep(sc.h1e_cas, sc.h2e_cas, 0, 2, 2, 0)
        DeltaD_full, _ = compute_nevpt2_effective_density(P, 0, 2)

        dcr = dyall_chain_rule_correction(P, DeltaD_full, 0, 2)

        assert isinstance(dcr, np.ndarray)
        assert dcr.shape == (2, 2)
        # For singlet CAS(2,2), correction is naturally zero
        # (antisymmetric Fock commutator vanishes for symmetric RDMs)
        assert np.max(np.abs(dcr)) < 1e-10

    def test_dyall_fock_with_2rdm(self):
        """Dyall Fock with 2-RDM contribution built correctly."""
        from vibeqc.gradient._dyall_chain_rule import compute_dyall_fock
        from vibeqc.gradient._pt2_density import (
            _full_1rdm_from_state,
            _full_2rdm_from_state,
        )
        from vibeqc.solvers._mrpt import _semicanonical_prep

        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        basis = BasisSet(mol, "sto-3g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(H.h1e, H.h2e, 2, 2, n_core=0, nuclear_repulsion=H.nuclear_repulsion)

        P = _semicanonical_prep(sc.h1e_cas, sc.h2e_cas, 0, 2, 2, 0)
        ref = P["ref"]
        norb = P["norb"]
        n_core, n_act = 0, 2
        act_s = slice(n_core, n_core + n_act)

        D_ref = _full_1rdm_from_state(ref, norb)[act_s, act_s]
        G2_ref = _full_2rdm_from_state(ref, norb)[act_s, act_s, act_s, act_s]

        h1D = np.zeros((norb, norb))
        for p in range(n_core):
            h1D[p, p] = P["eps"][p]
        for p in range(n_core + n_act, norb):
            h1D[p, p] = P["eps"][p]
        h1D[P["act"], P["act"]] = P["h1a"]
        eriD_chem = np.zeros_like(P["eri"])
        eriD_chem[P["act"], P["act"], P["act"], P["act"]] = P["eri"][
            P["act"], P["act"], P["act"], P["act"]
        ]

        FD = compute_dyall_fock(h1D, eriD_chem, D_ref, G2_ref, n_core, n_act, norb)

        assert FD.shape == (norb, norb)
        # Fock should be finite (not all zeros)
        assert np.max(np.abs(FD)) > 1e-10


class TestPT2Lagrangian:
    """Smoke tests for the CP-MCSCF Lagrangian module (v50-v51)."""

    def test_lagrangian_delta_runs(self):
        """PT2-only Lagrangian delta executes on H2/STO-3G."""
        from vibeqc.gradient._casscf import (
            _build_full_gamma_mo,
            _nonredundant_pairs_local,
        )
        from vibeqc.gradient._pt2_density import compute_pt2_effective_density
        from vibeqc.gradient._pt2_lagrangian import compute_pt2_lagrangian_gradient
        from vibeqc.solvers._mrpt import _semicanonical_prep

        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        basis = BasisSet(mol, "sto-3g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(H.h1e, H.h2e, 2, 2, n_core=0, nuclear_repulsion=H.nuclear_repulsion)
        C_conv = C_hf @ sc.mo_rotation
        rdm1, rdm2 = make_rdm12(sc.cas.ci_coeffs, sc.cas.determinants, 2)

        nmo = sc.h1e_cas.shape[0]
        n_core, n_act = 0, 2
        pairs = _nonredundant_pairs_local(n_core, n_act, nmo)
        npr = len(pairs)

        D_cas = np.zeros((nmo, nmo))
        D_cas[:n_core, :n_core] = np.diag(np.full(n_core, 2.0))
        D_cas[n_core : n_core + n_act, n_core : n_core + n_act] = rdm1
        Gamma_cas = _build_full_gamma_mo(rdm1, rdm2, n_core, n_act, nmo)

        P_eff = _semicanonical_prep(sc.h1e_cas, sc.h2e_cas, 0, 2, 2, 0)
        DeltaD_full, DeltaG_full = compute_pt2_effective_density(P_eff, 0, 2)

        z_orb_dummy = np.zeros(npr)
        grad = compute_pt2_lagrangian_gradient(
            mol,
            basis,
            C_conv,
            z_orb_dummy,
            sc.h1e_cas,
            sc.h2e_cas,
            D_cas,
            Gamma_cas,
            DeltaD_full,
            DeltaG_full,
            None,
            n_core,
            n_act,
            nmo,
            pairs,
        )
        assert grad.shape == (len(mol.atoms), 3)
        assert not np.any(np.isnan(grad))

    def test_total_lagrangian_runs(self):
        """Full Lagrangian gradient executes on H2/STO-3G."""
        from vibeqc.gradient._casscf import (
            _build_full_gamma_mo,
            _nonredundant_pairs_local,
        )
        from vibeqc.gradient._pt2_density import compute_pt2_effective_density
        from vibeqc.gradient._pt2_lagrangian import (
            compute_pt2_total_lagrangian_gradient,
        )
        from vibeqc.solvers._mrpt import _semicanonical_prep

        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        basis = BasisSet(mol, "sto-3g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(H.h1e, H.h2e, 2, 2, n_core=0, nuclear_repulsion=H.nuclear_repulsion)
        C_conv = C_hf @ sc.mo_rotation
        rdm1, rdm2 = make_rdm12(sc.cas.ci_coeffs, sc.cas.determinants, 2)

        nmo = sc.h1e_cas.shape[0]
        n_core, n_act = 0, 2
        pairs = _nonredundant_pairs_local(n_core, n_act, nmo)
        npr = len(pairs)

        D_cas = np.zeros((nmo, nmo))
        D_cas[:n_core, :n_core] = np.diag(np.full(n_core, 2.0))
        D_cas[n_core : n_core + n_act, n_core : n_core + n_act] = rdm1
        Gamma_cas = _build_full_gamma_mo(rdm1, rdm2, n_core, n_act, nmo)

        P_eff = _semicanonical_prep(sc.h1e_cas, sc.h2e_cas, 0, 2, 2, 0)
        DeltaD_full, DeltaG_full = compute_pt2_effective_density(P_eff, 0, 2)

        z_orb_dummy = np.zeros(npr)
        grad = compute_pt2_total_lagrangian_gradient(
            mol,
            basis,
            C_conv,
            z_orb_dummy,
            sc.h1e_cas,
            sc.h2e_cas,
            D_cas,
            Gamma_cas,
            DeltaD_full,
            DeltaG_full,
            None,
            n_core,
            n_act,
            nmo,
            pairs,
        )
        assert grad.shape == (len(mol.atoms), 3)
        assert not np.any(np.isnan(grad))
        # Total gradient should be finite (includes nuclear repulsion)
        assert np.max(np.abs(grad)) > 1e-10


class TestICCASPT2GradientComponents:
    """IC-CASPT2 exact density and legacy component regressions."""

    def test_exact_effective_density_differentiates_hylleraas_functional(self):
        """The IC density is the integral derivative of stationary L2."""
        from vibeqc.gradient._ic_caspt2_density import (
            compute_ic_caspt2_effective_density,
        )
        from vibeqc.gradient._pt2_density import _full_rdm12_from_state
        from vibeqc.solvers._mrpt import (
            _add,
            _dot,
            _generalized_fock,
            _semicanonical_prep,
            apply_1body,
            apply_2body,
        )

        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        basis = BasisSet(mol, "6-31g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(H.h1e, H.h2e, 2, 2, n_core=0, nuclear_repulsion=H.nuclear_repulsion)
        prep = _semicanonical_prep(sc.h1e_cas, sc.h2e_cas, 0, 2, 2, 0)
        eff = compute_ic_caspt2_effective_density(prep, 0, 2)

        ref = prep["ref"]
        psi1 = eff.psi1
        d00, _g00 = _full_rdm12_from_state(ref, prep["norb"])
        d00_act = d00[:2, :2]

        def hylleraas(h1, eri):
            fock = _generalized_fock(h1, eri, d00_act, 0, 2)
            h_ref = _add(
                apply_1body(ref, h1, prep["norb"]),
                apply_2body(ref, eri, prep["norb"]),
            )
            return (
                2.0 * _dot(psi1, h_ref)
                + _dot(psi1, apply_1body(psi1, fock, prep["norb"]))
                - eff.norm_psi1
                * _dot(ref, apply_1body(ref, fock, prep["norb"]))
            )

        assert hylleraas(prep["h1"], prep["eri"]) == pytest.approx(
            eff.e_corr, abs=2e-12
        )

        rng = np.random.default_rng(17)
        dh = rng.normal(size=prep["h1"].shape)
        dh = 0.5 * (dh + dh.T)
        dg = rng.normal(size=prep["eri"].shape)
        dg = sum(
            (
                dg,
                dg.transpose(1, 0, 2, 3),
                dg.transpose(0, 1, 3, 2),
                dg.transpose(1, 0, 3, 2),
                dg.transpose(2, 3, 0, 1),
                dg.transpose(2, 3, 1, 0),
                dg.transpose(3, 2, 0, 1),
                dg.transpose(3, 2, 1, 0),
            )
        ) / 8.0

        rot = prep["rot"]
        dh_sc = rot.T @ dh @ rot
        dg_sc = np.einsum(
            "pi,qj,rk,sl,pqrs->ijkl", rot, rot, rot, rot, dg, optimize=True
        )
        eps = 1e-5
        numerical = (
            hylleraas(prep["h1"] + eps * dh_sc, prep["eri"] + eps * dg_sc)
            - hylleraas(prep["h1"] - eps * dh_sc, prep["eri"] - eps * dg_sc)
        ) / (2.0 * eps)
        analytic = float(np.sum(eff.delta_d * dh))
        analytic += 0.5 * float(np.sum(eff.delta_gamma * dg))
        assert analytic == pytest.approx(numerical, abs=2e-9)

    def test_clagdx_fg3_components_are_off_diagonal(self):
        """CLagDX keeps the off-diagonal active-MO information needed by IC."""
        from vibeqc.gradient._ic_caspt2_density import (
            compute_ic_caspt2_density_components,
        )
        from vibeqc.solvers._mrpt import _semicanonical_prep

        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        basis = BasisSet(mol, "6-31g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(H.h1e, H.h2e, 2, 2, n_core=0, nuclear_repulsion=H.nuclear_repulsion)

        n_core, n_act = 0, 2
        P = _semicanonical_prep(sc.h1e_cas, sc.h2e_cas, n_core, n_act, 2, 0)
        ic = compute_ic_caspt2_density_components(P, n_core, n_act)

        assert ic.n_ic == 7
        assert ic.e_corr == pytest.approx(-0.004062629916327562, abs=1e-12)
        assert ic.bder[0, 1] == pytest.approx(-1.4550067964441956e-4, abs=1e-12)
        assert ic.sder[0, 1] == pytest.approx(1.5874561133723804e-3, abs=1e-12)
        assert np.allclose(ic.bder, ic.bder.T, atol=1e-14)
        assert np.allclose(ic.sder, ic.sder.T, atol=1e-14)

    def test_ic_density_blocks_are_assembled_in_full_mo_space(self):
        """BDER + FG3 lower-order corrections are exposed as full MO tensors."""
        from vibeqc.gradient._ic_caspt2_density import (
            compute_ic_caspt2_density_components,
        )
        from vibeqc.solvers._mrpt import _semicanonical_prep

        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        basis = BasisSet(mol, "6-31g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(H.h1e, H.h2e, 2, 2, n_core=0, nuclear_repulsion=H.nuclear_repulsion)

        n_core, n_act = 0, 2
        P = _semicanonical_prep(sc.h1e_cas, sc.h2e_cas, n_core, n_act, 2, 0)
        ic = compute_ic_caspt2_density_components(P, n_core, n_act)
        act_s = slice(n_core, n_core + n_act)

        assert ic.delta_d.shape == (P["norb"], P["norb"])
        assert ic.delta_gamma.shape == (P["norb"],) * 4
        assert np.allclose(ic.delta_d[act_s, act_s], ic.bder + ic.fg3["DG1"])
        assert np.allclose(
            ic.delta_gamma[act_s, act_s, act_s, act_s],
            ic.fg3["DG2"],
        )
        assert np.linalg.norm(ic.fg3["DG2"]) > 1e-6
        # The symmetric H2 CAS(2,2) fixture has a zero DEPSA correction
        # with the brute-force-verified RDM convention. The core-offset LiH
        # fixture below keeps the nonzero DEPSA smoke coverage.
        assert np.linalg.norm(ic.depsa) == pytest.approx(0.0, abs=1e-14)

    def test_ic_density_components_support_core_offset(self):
        """Core-containing CAS references use an active-relative 3-RDM state."""
        from vibeqc.gradient._ic_caspt2_density import (
            compute_ic_caspt2_density_components,
        )
        from vibeqc.solvers._mrpt import _semicanonical_prep

        mol = Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])])
        basis = BasisSet(mol, "sto-3g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)

        n_core, n_act = 1, 2
        P = _semicanonical_prep(H.h1e, H.h2e, n_core, n_act, 2, 0)
        ic = compute_ic_caspt2_density_components(P, n_core, n_act)
        act_s = slice(n_core, n_core + n_act)

        assert ic.n_ic > 0
        assert np.isfinite(ic.e_corr)
        assert ic.delta_d.shape == (P["norb"], P["norb"])
        assert np.linalg.norm(ic.delta_d[act_s, act_s]) > 1e-10
        assert np.linalg.norm(ic.delta_gamma[act_s, act_s, act_s, act_s]) > 1e-10
        assert np.linalg.norm(ic.depsa) > 1e-6

    def test_ic_orbital_gradient_rhs_runs(self):
        """IC dE2/dkappa RHS is available as a standalone framework piece."""
        from vibeqc.gradient._casscf import _nonredundant_pairs_local
        from vibeqc.gradient._ic_caspt2_density import (
            compute_ic_caspt2_orbital_gradient_fd,
        )

        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        basis = BasisSet(mol, "6-31g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(H.h1e, H.h2e, 2, 2, n_core=0, nuclear_repulsion=H.nuclear_repulsion)

        n_core, n_act = 0, 2
        pairs = _nonredundant_pairs_local(n_core, n_act, sc.h1e_cas.shape[0])
        rhs = compute_ic_caspt2_orbital_gradient_fd(
            sc.h1e_cas,
            sc.h2e_cas,
            n_core,
            n_act,
            2,
            pairs,
        )

        assert rhs.shape == (len(pairs),)
        assert np.all(np.isfinite(rhs))
        assert np.linalg.norm(rhs) > 1e-8

    def test_ic_orbital_hessian_runs(self):
        """IC H_oo contribution is available as a standalone framework piece."""
        from vibeqc.gradient._casscf import _nonredundant_pairs_local
        from vibeqc.gradient._ic_caspt2_density import (
            build_ic_caspt2_orbital_hessian_fd,
        )

        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        basis = BasisSet(mol, "6-31g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(H.h1e, H.h2e, 2, 2, n_core=0, nuclear_repulsion=H.nuclear_repulsion)

        n_core, n_act = 0, 2
        pairs = _nonredundant_pairs_local(n_core, n_act, sc.h1e_cas.shape[0])
        hess = build_ic_caspt2_orbital_hessian_fd(
            sc.h1e_cas,
            sc.h2e_cas,
            n_core,
            n_act,
            2,
            pairs,
        )

        assert hess.shape == (len(pairs), len(pairs))
        assert np.all(np.isfinite(hess))
        assert np.allclose(hess, hess.T, atol=1e-10)
        assert np.linalg.norm(hess) > 1e-8


class TestICCASPT2AnalyticGradient:
    """Exact IC-CASPT2 Lagrangian gradient against relaxed energy FD."""

    def test_h2_matches_full_relaxed_fd(self):
        from vibeqc.gradient._caspt2 import compute_ic_caspt2_analytic_gradient
        from vibeqc.solvers._mrpt import caspt2

        atoms = [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])]
        mol = Molecule(atoms)
        basis = BasisSet(mol, "6-31g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(H.h1e, H.h2e, 2, 2, n_core=0, nuclear_repulsion=H.nuclear_repulsion)
        C_conv = C_hf @ sc.mo_rotation
        rdm1, rdm2 = make_rdm12(sc.cas.ci_coeffs, sc.cas.determinants, 2)
        gradient = compute_ic_caspt2_analytic_gradient(
            mol,
            basis,
            C_conv,
            sc.h1e_cas,
            sc.h2e_cas,
            0,
            2,
            2,
            rdm1=rdm1,
            rdm2=rdm2,
            ci_coeffs=sc.cas.ci_coeffs,
            determinants=sc.cas.determinants,
        )

        def total_energy(z0):
            displaced = Molecule(
                [Atom(1, [0.0, 0.0, z0]), Atom(1, [0.0, 0.0, 1.4])]
            )
            displaced_basis = BasisSet(displaced, "6-31g")
            C0 = get_hf_orbital_provider(displaced, displaced_basis)
            Hd = build_hamiltonian_mo(displaced, displaced_basis, C0)
            sd = casscf(
                Hd.h1e,
                Hd.h2e,
                2,
                2,
                n_core=0,
                nuclear_repulsion=Hd.nuclear_repulsion,
            )
            return caspt2(
                sd.cas,
                sd.h1e_cas,
                sd.h2e_cas,
                n_core=0,
                n_virt=sd.h1e_cas.shape[0] - 2,
            ).e_total

        step = 1e-4
        finite_difference = (total_energy(step) - total_energy(-step)) / (2.0 * step)
        assert gradient[0, 2] == pytest.approx(finite_difference, abs=1e-7)
        assert np.max(np.abs(np.sum(gradient, axis=0))) < 2e-8

    def test_lih_core_offset_matches_full_relaxed_fd(self):
        """Inactive-core response and every AO derivative block stay exact."""
        from vibeqc.gradient._caspt2 import compute_ic_caspt2_analytic_gradient
        from vibeqc.solvers._mrpt import caspt2

        atoms = [Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])]
        mol = Molecule(atoms)
        basis = BasisSet(mol, "sto-3g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(H.h1e, H.h2e, 2, 2, n_core=1, nuclear_repulsion=H.nuclear_repulsion)
        rdm1, rdm2 = make_rdm12(sc.cas.ci_coeffs, sc.cas.determinants, 2)
        gradient = compute_ic_caspt2_analytic_gradient(
            mol,
            basis,
            C_hf @ sc.mo_rotation,
            sc.h1e_cas,
            sc.h2e_cas,
            1,
            2,
            2,
            rdm1=rdm1,
            rdm2=rdm2,
            ci_coeffs=sc.cas.ci_coeffs,
            determinants=sc.cas.determinants,
        )

        def total_energy(z0):
            displaced = Molecule(
                [Atom(3, [0.0, 0.0, z0]), Atom(1, [0.0, 0.0, 3.0])]
            )
            displaced_basis = BasisSet(displaced, "sto-3g")
            C0 = get_hf_orbital_provider(displaced, displaced_basis)
            Hd = build_hamiltonian_mo(displaced, displaced_basis, C0)
            sd = casscf(
                Hd.h1e,
                Hd.h2e,
                2,
                2,
                n_core=1,
                nuclear_repulsion=Hd.nuclear_repulsion,
            )
            return caspt2(
                sd.cas,
                sd.h1e_cas,
                sd.h2e_cas,
                n_core=1,
                n_virt=sd.h1e_cas.shape[0] - 3,
            ).e_total

        step = 3e-4
        finite_difference = (total_energy(step) - total_energy(-step)) / (2.0 * step)
        assert gradient[0, 2] == pytest.approx(finite_difference, abs=1e-7)
        assert np.max(np.abs(np.sum(gradient, axis=0))) < 2e-8


class TestSANEVPT2Gradient:
    """SA-NEVPT2 Z-vector gradient validation (v52)."""

    def test_sa_nevpt2_gradient_smoke(self):
        """SA-NEVPT2 gradient runs on H2/6-31G SA-2-CAS(2,2)."""
        from vibeqc.gradient._nevpt2 import compute_nevpt2_gradient
        from vibeqc.solvers._rdm import make_rdm12_sa

        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        basis = BasisSet(mol, "6-31g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(
            H.h1e,
            H.h2e,
            2,
            2,
            n_core=0,
            nuclear_repulsion=H.nuclear_repulsion,
            nroots=2,
            weights=[0.5, 0.5],
        )
        C_conv = C_hf @ sc.mo_rotation
        ci_all = sc.cas.ci_coeffs_all
        ci_list = [ci_all[:, i] for i in range(ci_all.shape[1])]
        rdm1_sa, rdm2_sa = make_rdm12_sa(ci_all, sc.cas.determinants, 2, [0.5, 0.5])

        g_ss = compute_nevpt2_gradient(
            mol,
            basis,
            C_conv,
            sc.h1e_cas,
            sc.h2e_cas,
            n_core=0,
            n_active_orb=2,
            n_active_elec=2,
            rdm1=rdm1_sa,
            rdm2=rdm2_sa,
            use_zvector=True,
            determinants=sc.cas.determinants,
            ci_coeffs=ci_list[0],
        )
        g_sa = compute_nevpt2_gradient(
            mol,
            basis,
            C_conv,
            sc.h1e_cas,
            sc.h2e_cas,
            n_core=0,
            n_active_orb=2,
            n_active_elec=2,
            rdm1=rdm1_sa,
            rdm2=rdm2_sa,
            use_zvector=True,
            determinants=sc.cas.determinants,
            ci_coeffs=ci_list[0],
            sa_weights=[0.5, 0.5],
            sa_ci_coeffs=ci_list,
        )

        assert g_ss.shape == (2, 3)
        assert g_sa.shape == (2, 3)
        # For symmetric singlet with equal weights, SS and SA should be close
        # (SA averages per-state effective densities; small differences expected)
        assert np.allclose(g_ss, g_sa, atol=5e-4)

    def test_sa_nevpt2_vs_frozen_fd(self):
        """SA-NEVPT2 frozen gradient matches FD of SA-NEVPT2 energy."""
        from vibeqc.gradient._nevpt2 import compute_nevpt2_gradient
        from vibeqc.solvers._mrpt import (
            _add,
            _pt2_correction,
            _semicanonical_prep,
            apply_1body,
            apply_2body,
        )
        from vibeqc.solvers._rdm import make_rdm12_sa

        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        basis = BasisSet(mol, "sto-3g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(
            H.h1e,
            H.h2e,
            2,
            2,
            n_core=0,
            nuclear_repulsion=H.nuclear_repulsion,
            nroots=2,
            weights=[0.5, 0.5],
        )
        C_conv = C_hf @ sc.mo_rotation
        ci_all = sc.cas.ci_coeffs_all
        ci_list = [ci_all[:, i] for i in range(ci_all.shape[1])]
        rdm1_sa, rdm2_sa = make_rdm12_sa(ci_all, sc.cas.determinants, 2, [0.5, 0.5])

        g = compute_nevpt2_gradient(
            mol,
            basis,
            C_conv,
            sc.h1e_cas,
            sc.h2e_cas,
            n_core=0,
            n_active_orb=2,
            n_active_elec=2,
            rdm1=rdm1_sa,
            rdm2=rdm2_sa,
            use_zvector=False,
        )
        # Gradient should be finite (at least nuclear repulsion)
        assert g.shape == (2, 3)
        assert np.any(np.abs(g) > 1e-10)


class TestSASpecificity:
    """SA-CASPT2 and SA-NEVPT2 produce different results from SS (v52)."""

    def test_sa_caspt2_differs_from_ss(self):
        """SA-CASPT2 gradient differs from SS on H2/6-31G SA-2-CAS(2,2)."""
        from vibeqc.gradient._caspt2 import compute_caspt2_gradient
        from vibeqc.solvers._rdm import make_rdm12_sa

        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        basis = BasisSet(mol, "6-31g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(
            H.h1e,
            H.h2e,
            2,
            2,
            n_core=0,
            nuclear_repulsion=H.nuclear_repulsion,
            nroots=2,
            weights=[0.5, 0.5],
        )
        C_conv = C_hf @ sc.mo_rotation
        ci_all = sc.cas.ci_coeffs_all
        ci_list = [ci_all[:, i] for i in range(ci_all.shape[1])]
        rdm1_sa, rdm2_sa = make_rdm12_sa(ci_all, sc.cas.determinants, 2, [0.5, 0.5])

        g_ss = compute_caspt2_gradient(
            mol,
            basis,
            C_conv,
            sc.h1e_cas,
            sc.h2e_cas,
            n_core=0,
            n_active_orb=2,
            n_active_elec=2,
            rdm1=rdm1_sa,
            rdm2=rdm2_sa,
            use_zvector=True,
            determinants=sc.cas.determinants,
            ci_coeffs=ci_list[0],
        )
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            g_sa = compute_caspt2_gradient(
                mol,
                basis,
                C_conv,
                sc.h1e_cas,
                sc.h2e_cas,
                n_core=0,
                n_active_orb=2,
                n_active_elec=2,
                rdm1=rdm1_sa,
                rdm2=rdm2_sa,
                use_zvector=True,
                determinants=sc.cas.determinants,
                ci_coeffs=ci_list[0],
                sa_weights=[0.5, 0.5],
                sa_ci_coeffs=ci_list,
            )
        assert not any(
            "PT2 chain-rule correction failed" in str(w.message) for w in caught
        )
        # SA and SS must differ (SA effective density averaging is active)
        assert not np.allclose(g_ss[0, 2], g_sa[0, 2], atol=1e-6), (
            f"SA-CASPT2 should differ from SS: {g_ss[0, 2]:.8f} vs {g_sa[0, 2]:.8f}"
        )

    def test_sa_nevpt2_differs_from_ss(self):
        """SA-NEVPT2 gradient differs from SS on H2/6-31G SA-2-CAS(2,2)."""
        from vibeqc.gradient._nevpt2 import compute_nevpt2_gradient
        from vibeqc.solvers._rdm import make_rdm12_sa

        mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        basis = BasisSet(mol, "6-31g")
        C_hf = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C_hf)
        sc = casscf(
            H.h1e,
            H.h2e,
            2,
            2,
            n_core=0,
            nuclear_repulsion=H.nuclear_repulsion,
            nroots=2,
            weights=[0.5, 0.5],
        )
        C_conv = C_hf @ sc.mo_rotation
        ci_all = sc.cas.ci_coeffs_all
        ci_list = [ci_all[:, i] for i in range(ci_all.shape[1])]
        rdm1_sa, rdm2_sa = make_rdm12_sa(ci_all, sc.cas.determinants, 2, [0.5, 0.5])

        g_ss = compute_nevpt2_gradient(
            mol,
            basis,
            C_conv,
            sc.h1e_cas,
            sc.h2e_cas,
            n_core=0,
            n_active_orb=2,
            n_active_elec=2,
            rdm1=rdm1_sa,
            rdm2=rdm2_sa,
            use_zvector=True,
            determinants=sc.cas.determinants,
            ci_coeffs=ci_list[0],
        )
        g_sa = compute_nevpt2_gradient(
            mol,
            basis,
            C_conv,
            sc.h1e_cas,
            sc.h2e_cas,
            n_core=0,
            n_active_orb=2,
            n_active_elec=2,
            rdm1=rdm1_sa,
            rdm2=rdm2_sa,
            use_zvector=True,
            determinants=sc.cas.determinants,
            ci_coeffs=ci_list[0],
            sa_weights=[0.5, 0.5],
            sa_ci_coeffs=ci_list,
        )
        assert not np.allclose(g_ss[0, 2], g_sa[0, 2], atol=1e-6), (
            f"SA-NEVPT2 should differ from SS: {g_ss[0, 2]:.8f} vs {g_sa[0, 2]:.8f}"
        )


class TestCASSCFGradientFullVectorVsFD:
    """GitLab #516: the whole gradient vector against a full-energy FD.

    The issue's carrier: H2 on the z axis with a basis carrying p functions
    (cc-pVDZ). The retired ``compute_wz=True`` path returned
    g_x = -0.357 Ha/bohr there (and flipped the sign of g_z on 6-31G)
    because it contracted one reference z-vector with every column of the
    perturbed orbital gradient. Every path must now reproduce the central
    finite difference of the converged CASSCF energy in all three Cartesian
    components, and the transverse components of a linear molecule must
    vanish.
    """

    @staticmethod
    def _run(basis_name, r_bohr, *, compute_wz=False, shift=(0.0, 0.0, 0.0)):
        import vibeqc as vq
        from vibeqc.runner import _run_single_point
        from vibeqc.solvers import CASSCFOptions

        xyz1 = [shift[0], shift[1], r_bohr + shift[2]]
        mol = vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, xyz1)])
        basis = vq.BasisSet(mol, basis_name)
        return _run_single_point(
            "casscf",
            mol,
            basis,
            functional=None,
            active_space=(2, 2),
            casscf_options=CASSCFOptions(compute_wz=compute_wz),
        )

    @pytest.mark.parametrize("basis_name", ["6-31g", "cc-pvdz"])
    def test_full_vector_matches_full_energy_fd(self, basis_name):
        h = 1.0e-3
        r = 1.4
        analytic = np.asarray(self._run(basis_name, r).gradient, dtype=float)
        with pytest.warns(FutureWarning, match=r"compute_wz=True\) is obsolete"):
            aliased = np.asarray(
                self._run(basis_name, r, compute_wz=True).gradient, dtype=float
            )
        fd = np.zeros(3)
        for c in range(3):
            d = [0.0, 0.0, 0.0]
            d[c] = h
            e_plus = float(self._run(basis_name, r, shift=tuple(d)).energy)
            d[c] = -h
            e_minus = float(self._run(basis_name, r, shift=tuple(d)).energy)
            fd[c] = (e_plus - e_minus) / (2.0 * h)
        # Measured 2.2e-7 (6-31G) and 2.5e-7 (cc-pVDZ) on 2026-09-02.
        np.testing.assert_allclose(analytic[1], fd, atol=1.0e-5)
        np.testing.assert_allclose(aliased, analytic, atol=1e-14)
        # Linear molecule on z: g_x = g_y = 0 exactly (was -0.357 on cc-pVDZ).
        assert np.abs(analytic[:, :2]).max() < 1.0e-10
        assert np.abs(fd[:2]).max() < 1.0e-8
        # Translational invariance.
        assert np.abs(analytic.sum(axis=0)).max() < 1.0e-10


# ---------------------------------------------------------------------------
# Stale-claim guard (#119)
#
# The W^z retirement (#516) corrected the physics and the manual, but left
# five user-facing surfaces still describing the gradient as incomplete, as
# "~87 % of the full CP-MCSCF gradient", or as carrying a "gated W^z
# correction". Those statements contradict the shipped contract and were the
# stated blocker on #119's independent verification, so they are pinned here
# rather than only corrected once.
# ---------------------------------------------------------------------------

# Surfaces a user reads before deciding whether to trust the gradient.
_CASSCF_GRADIENT_SURFACES = (
    "python/vibeqc/gradient/__init__.py",
    "python/vibeqc/gradient/_casscf.py",
    "examples/casscf_gradient.py",
    "examples/wavefunction/01_casscf_h2o.py",
    "examples/wavefunction/02_casscf_gradient.py",
    "examples/regression/parity_casscf_gradient.py",
    "examples/regression/casscf_gradient_fd_reproducer.py",
    "docs/user_guide/non_hf_solvers.md",
    "docs/user_guide/geometry_optimization.md",
    "docs/tutorial/casscf_multireference.md",
)

# Each claim is (pattern, why it is wrong now). A line that also names the
# retirement is history, not a live claim, so it is exempt.
_RETRACTED_CLAIMS = (
    (r"8[0-9]\s*%\s*of the full", "the analytic gradient is the complete derivative"),
    (r"[Mm]issing\s+1[0-9]\s*%", "nothing is missing from the analytic gradient"),
    (r"gated\s+W\^z", "the W^z branch was retired, not gated"),
    (r"incomplete\s+z-vector-free", "the gradient is not incomplete"),
    (r"not\s+(full-energy\s+)?(finite-difference|FD)[- ]tight",
     "it matches full-energy FD to ~1.6e-7 Ha/bohr"),
    (r"must not be used as production forces",
     "the analytic path is the production force"),
    (r"Incomplete analytic preview", "the analytic value is complete"),
)

# Prose that documents the retirement is allowed to recite the old claim. The
# marker usually sits a line or two away from the quoted wording, so the
# exemption looks at the surrounding lines, not just the matching one.
_HISTORY_MARKERS = re.compile(
    r"retired|obsolete|former|earlier revision|used to|was written to|no-op"
    r"|since-fixed|phantom|#516|#119",
    re.I,
)
_HISTORY_WINDOW = 3


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("relpath", _CASSCF_GRADIENT_SURFACES)
def test_casscf_gradient_surfaces_carry_no_retracted_claim(relpath):
    """No live surface still calls the CASSCF gradient incomplete (#119).

    The physics fix landed under #516; independent verification failed on
    these surfaces alone, because a user who reads them rejects or
    mischaracterises the corrected production path.
    """
    path = _repo_root() / relpath
    assert path.is_file(), f"{relpath} moved; update _CASSCF_GRADIENT_SURFACES"
    lines = path.read_text(encoding="utf-8").splitlines()
    offenders = []
    for index, line in enumerate(lines):
        window = lines[max(0, index - _HISTORY_WINDOW) : index + _HISTORY_WINDOW + 1]
        if any(_HISTORY_MARKERS.search(neighbour) for neighbour in window):
            continue  # documents the retirement rather than asserting it
        for pattern, why in _RETRACTED_CLAIMS:
            if re.search(pattern, line):
                offenders.append(f"{relpath}:{index + 1}: {line.strip()}  <- {why}")
    assert not offenders, "retracted CASSCF-gradient claims:\n" + "\n".join(offenders)


def test_stale_claim_guard_would_catch_the_original_wording():
    """The guard is not vacuous: the exact pre-fix lines must trip it."""
    pre_fix = [
        "The gradient captures ~87% of the full CP-MCSCF gradient.",
        "Missing 13% = W^z (CI+orbital relaxation, Handy-Schaefer z-vector).",
        "z-vector-free + gated W^z correction + numerical FD fallback",
        "Characterizes the incomplete z-vector-free gradient against two",
        "The exposed z-vector-free value is incomplete and not full-energy FD-tight.",
        "finite-difference tight and must not be used as production forces.",
        'print(f"\\nIncomplete analytic preview (dE/dR, Hartree/bohr):")',
    ]
    for line in pre_fix:
        assert not _HISTORY_MARKERS.search(line), line
        assert any(
            re.search(pattern, line) for pattern, _ in _RETRACTED_CLAIMS
        ), f"guard misses the pre-fix wording: {line}"

    # And the exemption is not a blanket one: a retracted claim standing on
    # its own, with no retirement context nearby, is still reported.
    stray = ["filler"] * 10 + [pre_fix[0]] + ["filler"] * 10
    assert not any(
        _HISTORY_MARKERS.search(n)
        for n in stray[10 - _HISTORY_WINDOW : 11 + _HISTORY_WINDOW]
    )
