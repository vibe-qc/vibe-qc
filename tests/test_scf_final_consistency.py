"""Audit regression: SCF result.energy is consistent with result.density /
result.fock after the converged-branch final-rebuild pass.

Pre-fix the four drivers set ``result.energy = E_total`` *before* the
converged-branch rebuild of (F_final, C_final, D), so the reported
energy described E(D_used, F_pre-rebuild), not E(F_final, D). At loose
SCF tolerance this gap was ~1e-3 Ha (2026-05-18 audit probe).

Each test forces a deliberately loose convergence so the gap, if it
returned, would be ~1e-3 Ha rather than uHa, well outside the
1e-12 Ha tolerance these tests assert.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    InitialGuess,
    Molecule,
    RHFOptions,
    RKSOptions,
    UHFOptions,
    UKSOptions,
    compute_kinetic,
    compute_nuclear,
    compute_overlap,
    run_rhf,
    run_rks,
    run_uhf,
    run_uks,
)


def _e_nuc_classical(mol: Molecule) -> float:
    atoms = mol.atoms
    e = 0.0
    for i, a in enumerate(atoms):
        for j in range(i + 1, len(atoms)):
            b = atoms[j]
            r = np.sqrt(sum((a.xyz[c] - b.xyz[c]) ** 2 for c in range(3)))
            e += a.Z * b.Z / r
    return e


def _h2o():
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, 1.81]),
            Atom(1, [1.71, 0.0, -0.45]),
        ],
        0,
        1,
    )


def _oh():
    return Molecule(
        [Atom(8, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.81])], 0, 2
    )


def _hcore(mol: Molecule, basis: BasisSet) -> np.ndarray:
    return np.asarray(compute_kinetic(basis)) + np.asarray(
        compute_nuclear(basis, mol)
    )


def _assert_uks_final_density_consistent(result, basis, opts, occupations):
    """Check the determinant and stationarity that ``converged`` promises."""
    overlap = np.asarray(compute_overlap(basis))
    overlap_eigenvalues, overlap_eigenvectors = np.linalg.eigh(overlap)
    keep = overlap_eigenvalues > opts.linear_dep_threshold
    orthogonalizer = (
        overlap_eigenvectors[:, keep] / np.sqrt(overlap_eigenvalues[keep])
    )
    returned_norms = []
    for spin, n_occ in occupations:
        fock = np.asarray(getattr(result, f"fock_{spin}"))
        density = np.asarray(getattr(result, f"density_{spin}"))
        coefficients = np.asarray(getattr(result, f"mo_coeffs_{spin}"))
        error_ao = fock @ density @ overlap - overlap @ density @ fock
        returned_norms.append(
            np.linalg.norm(orthogonalizer.T @ error_ao @ orthogonalizer)
        )
        assert np.linalg.norm(density @ overlap @ density - density) < 1.0e-12
        mo_projector = (
            coefficients[:, :n_occ] @ coefficients[:, :n_occ].T
        )
        occupied_overlap = np.trace(
            density @ overlap @ mo_projector @ overlap
        )
        assert occupied_overlap == pytest.approx(n_occ, abs=1.0e-5)

    assert max(returned_norms) <= opts.conv_tol_grad
    assert abs(result.energy - result.scf_trace[-1].energy) < (
        opts.conv_tol_energy
    )


def test_rhf_energy_matches_final_F_D_at_loose_tol():
    mol = _h2o()
    basis = BasisSet(mol, "sto-3g")
    opts = RHFOptions()
    opts.conv_tol_energy = 1e-3
    opts.conv_tol_grad = 1e-3
    r = run_rhf(mol, basis, opts)
    assert r.converged
    Hcore = _hcore(mol, basis)
    F = np.asarray(r.fock)
    D = np.asarray(r.density)
    E_elec = 0.5 * float((D * (Hcore + F)).sum())
    E_check = E_elec + _e_nuc_classical(mol)
    assert r.energy == pytest.approx(E_check, abs=1e-12)


def test_uhf_energy_matches_final_F_D_at_loose_tol():
    mol = _oh()
    basis = BasisSet(mol, "sto-3g")
    opts = UHFOptions()
    opts.conv_tol_energy = 1e-3
    opts.conv_tol_grad = 1e-3
    r = run_uhf(mol, basis, opts)
    assert r.converged
    Hcore = _hcore(mol, basis)
    Fa = np.asarray(r.fock_alpha)
    Fb = np.asarray(r.fock_beta)
    Da = np.asarray(r.density_alpha)
    Db = np.asarray(r.density_beta)
    E_elec = 0.5 * (
        float((Da * (Hcore + Fa)).sum()) + float((Db * (Hcore + Fb)).sum())
    )
    E_check = E_elec + _e_nuc_classical(mol)
    assert r.energy == pytest.approx(E_check, abs=1e-12)


def test_rks_energy_matches_decomposition_at_loose_tol():
    """RKS energy decomposition (e_coulomb / e_hf_exchange / e_xc /
    e_nuclear) must sum to ``result.energy`` plus the one-electron
    piece on the *final* density."""
    mol = _h2o()
    basis = BasisSet(mol, "sto-3g")
    opts = RKSOptions()
    opts.functional = "LDA"
    opts.conv_tol_energy = 1e-3
    opts.conv_tol_grad = 1e-3
    r = run_rks(mol, basis, opts)
    assert r.converged
    Hcore = _hcore(mol, basis)
    D = np.asarray(r.density)
    E_core = float((D * Hcore).sum())
    total = (
        E_core
        + r.e_coulomb
        + r.e_hf_exchange
        + r.e_xc
        + _e_nuc_classical(mol)
    )
    assert r.energy == pytest.approx(total, abs=1e-12)


def test_uks_energy_matches_decomposition_at_loose_tol():
    mol = _oh()
    basis = BasisSet(mol, "sto-3g")
    opts = UKSOptions()
    opts.functional = "LDA"
    opts.conv_tol_energy = 1e-3
    opts.conv_tol_grad = 1e-3
    r = run_uks(mol, basis, opts)
    assert r.converged
    Hcore = _hcore(mol, basis)
    D_total = np.asarray(r.density_alpha) + np.asarray(r.density_beta)
    E_core = float((D_total * Hcore).sum())
    total = (
        E_core
        + r.e_coulomb
        + r.e_hf_exchange
        + r.e_xc
        + _e_nuc_classical(mol)
    )
    assert r.energy == pytest.approx(total, abs=1e-12)


def test_uks_converged_result_density_clears_commutator_gate():
    """``converged`` certifies the density returned to the caller.

    The UKS loop used to test ``D_n`` and then return the untested
    ``D_{n+1}`` produced by its final extrapolated-Fock diagonalization.  On
    this compact SH/LDA witness the declaring trace row was
    7.939030700e-4, but the returned F/D pair was 9.136884917e-4: 14.21%
    above the requested 8e-4 stationarity gate.
    """
    angstrom_to_bohr = 1.0 / 0.529177210903
    mol = Molecule(
        [
            Atom(16, [0.0, 0.0, 0.0]),
            Atom(1, [1.341 * angstrom_to_bohr, 0.0, 0.0]),
        ],
        0,
        2,
    )
    basis = BasisSet(mol, "sto-3g")
    opts = UKSOptions()
    opts.functional = "LDA"
    opts.damping = 0.0
    opts.max_iter = 50
    opts.conv_tol_energy = 1.0e-6
    opts.conv_tol_grad = 8.0e-4
    opts.stability_check = False

    result = run_uks(mol, basis, opts)

    assert result.converged
    assert result.n_iter == 6

    _assert_uks_final_density_consistent(
        result, basis, opts, (("alpha", 9), ("beta", 8))
    )


def test_uks_damping_does_not_return_mixed_density():
    """A no-DIIS convergence cannot expose the damped work density."""
    mol = Molecule(
        [Atom(2, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])], 0, 2
    )
    basis = BasisSet(mol, "sto-3g")
    opts = UKSOptions()
    opts.functional = "LDA"
    # Keep the deliberately capped trajectory that originally exposed this bug.
    opts.initial_guess = InitialGuess.PATOM
    opts.use_diis = False
    opts.dynamic_damping = False
    opts.damping = 0.5
    opts.max_iter = 21
    opts.conv_tol_energy = 1.0e-8
    opts.conv_tol_grad = 1.0e-6
    opts.stability_check = False
    opts.auto_level_shift_on_oscillation = False
    opts.restart_opts.enabled = False

    result = run_uks(mol, basis, opts)

    assert result.converged
    assert result.n_iter == 21
    candidate_energy_change = result.scf_trace[-1].energy - result.energy
    assert 0.0 < candidate_energy_change < opts.conv_tol_energy
    _assert_uks_final_density_consistent(
        result, basis, opts, (("alpha", 2), ("beta", 1))
    )
