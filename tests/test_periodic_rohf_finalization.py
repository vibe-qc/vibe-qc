"""Small matrix contracts for the Gamma ROHF result assembly.

The integral/guess and eigensolver boundaries are replaced. The driver body and its
Roothaan occupation, density and effective-Fock helpers are loaded from source,
so this file can also run with ``python tests/test_periodic_rohf_finalization.py``
without a native build. These are algebra controls, not chemistry acceptance.
"""
from __future__ import annotations

import ast
from contextlib import nullcontext
from dataclasses import dataclass, field, replace, fields
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class _WithoutImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None

    def visit_Import(self, node):
        return None


def _source_functions(path, namespace, names=None):
    tree = ast.parse((ROOT / path).read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and (names is None or node.name in names)]
    tree = _WithoutImports().visit(tree)
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(ROOT / path), "exec"), namespace)


def _fixture_diagonalizer(x, overlap, options, **metadata):
    """Dense matrix oracle; shared-solver behavior has separate coverage."""
    def solve(fock):
        fock = getattr(fock, "matrix", fock)
        projected = x.T @ fock @ x
        energies, coefficients = np.linalg.eigh((projected + projected.T) * .5)
        return x @ coefficients, energies
    return solve


def _fixture_dense_fock(fock, *, X=None, overlap=None, **metadata):
    x = np.eye(fock.shape[0]) if X is None else X
    coefficients, energies = _fixture_diagonalizer(x, overlap, None)(fock)
    return SimpleNamespace(eigenvectors=coefficients, eigenvalues=energies)


def _matrix_run(*, open_shell=False, rank_reduced=False, max_iter=4,
                shift=0.2, nonlinear=False):
    """Run the unchanged driver control flow with at most four AO directions."""
    size = 4 if open_shell else 3
    overlap = np.diag([1.0] * (size - 1) + [1e-12 if rank_reduced else 1.0])
    hcore = np.diag([-2.0, 0.4, 1.4, 0.8] if open_shell else [-0.6, 0.7, 1.0])
    exchange = np.diag([0.0, 0.0, 0.6, 0.0] if open_shell else [0.0] * size)
    perturbation = np.zeros((size, size))
    perturbation[0, 1] = perturbation[1, 0] = 0.1 if nonlinear else 0.0
    n_alpha, n_beta = (2, 1) if open_shell else (1, 1)
    system = SimpleNamespace(
        dim=3, multiplicity=n_alpha - n_beta + 1, lattice=np.eye(3),
        n_electrons=lambda: n_alpha + n_beta, unit_cell_molecule=lambda: None,
    )
    options = SimpleNamespace(
        initial_guess="hcore", lattice_opts=SimpleNamespace(coulomb_method="ewald", cutoff_bohr=1.0),
        damping=0.0, use_diis=False, diis_start_iter=2, level_shift=shift,
        max_iter=max_iter, conv_tol_energy=1e-12, conv_tol_grad=1e-10,
    )
    logger = SimpleNamespace(**{name: lambda *a, **k: None for name in
                               ("info", "banner", "iteration", "write_raw")})
    logger.stage = lambda *a, **k: nullcontext()
    j_inputs = []

    def hartree(density):
        j_inputs.append(density.copy())
        return (0.2 * np.diag(np.diag(density)) + perturbation
                if nonlinear else np.zeros_like(density))

    def orthogonalizer(s, threshold, **kwargs):
        values, vectors = np.linalg.eigh(s)
        keep = values > threshold
        return vectors[:, keep] / np.sqrt(values[keep]), int(keep.sum())

    namespace = dict(
        np=np, dataclass=dataclass, field=field, replace=replace, fields=fields,
        hashlib=hashlib, json=json, math=math, __name__=__name__,
        _gamma_source_details=lambda *a, **k: dict(shells=[], ordering="synthetic"),
        CoulombMethod=SimpleNamespace(EWALD_3D="ewald"),
        InitialGuess=SimpleNamespace(PATOM="patom", SAD="sad"),
        select_periodic_driver_guess=lambda *a, **k: SimpleNamespace(effective="hcore"),
        periodic_result_selection=lambda *a, **k: None,
        resolve_progress=lambda *a, **k: logger,
        _refuse_if_dense_ionic=lambda *a: None,
        compute_overlap_lattice=lambda *a: overlap,
        compute_kinetic_lattice=lambda *a: hcore,
        compute_nuclear_lattice_dispatch=lambda *a: np.zeros_like(hcore),
        bloch_sum=lambda matrix, k: matrix,
        scf_preflight_overlap_check=lambda *a, **k: None,
        _canonical_orthogonalizer=orthogonalizer,
        FockDiagonalizer=_fixture_diagonalizer,
        solve_dense_fock=_fixture_dense_fock,
        nuclear_repulsion_per_cell=lambda *a: 0.0,
        initial_densities_open_shell=lambda *a, **k: None,
        make_ewald_3d_gamma_j_builder=lambda *a, **k: hartree,
        build_jk_gamma_molecular_limit=lambda b, sy, lo, d, t:
            SimpleNamespace(K=exchange * np.trace(d)),
        _madelung_energy_correction_for_lat=lambda *a: 0.0,
        check_scf_divergence=lambda *a: None,
        MOLECULAR_LIMIT="molecular-limit",
    )
    _source_functions("python/vibeqc/solvers/operators.py", namespace)
    _source_functions("python/vibeqc/fock_state.py", namespace)
    _source_functions("python/vibeqc/periodic_uhf_ewald.py", namespace, {"_GammaUHFBuilder"})
    _source_functions("python/vibeqc/rohf.py", namespace, {
        "roothaan_effective_fock", "_roothaan_occupations",
        "_densities_from_occupations", "_commutator_error",
    })
    namespace["SCFIteration"] = lambda **kwargs: SimpleNamespace(**kwargs)
    _source_functions("python/vibeqc/periodic_rohf_ewald.py", namespace)
    result = namespace["run_rohf_periodic_gamma_ewald3d"](
        system, SimpleNamespace(nbasis=size), options, omega=0.5, grid_shape=(1, 1, 1),
        auto_optimize_truncation=False,
    )
    return result, namespace, hcore, exchange, j_inputs


class GammaROHFFinalization(unittest.TestCase):
    def assert_consistent_state(self, result, ns, hcore, exchange, j_inputs):
        c, occ, s = result.mo_coeffs, result.mo_occupations, result.overlap
        da = (c * (occ > 0)) @ c.T
        db = (c * (occ == 2)) @ c.T
        np.testing.assert_allclose(da, result.density_alpha, atol=1e-12)
        np.testing.assert_allclose(db, result.density_beta, atol=1e-12)
        np.testing.assert_allclose(da + db, result.density, atol=1e-12)
        np.testing.assert_allclose(j_inputs[-1], result.density, atol=1e-12)
        np.testing.assert_allclose(c.T @ s @ c, np.eye(c.shape[1]), atol=1e-12)
        self.assertAlmostEqual(float(np.trace(da @ s)), result.n_alpha)
        self.assertAlmostEqual(float(np.trace(db @ s)), result.n_beta)
        f = ns["roothaan_effective_fock"](result.fock_alpha, result.fock_beta, da, db, s)
        np.testing.assert_allclose(f, result.fock, atol=1e-12)
        projected = c.T @ f @ c
        np.testing.assert_allclose(np.diag(projected), result.mo_energies, atol=1e-12)
        # Each same-occupation block is canonical, even before convergence.
        for occupation in (0.0, 1.0, 2.0):
            indices = np.flatnonzero(occ == occupation)
            block = projected[np.ix_(indices, indices)]
            np.testing.assert_allclose(block, np.diag(np.diag(block)), atol=1e-12)
        expected_energy = 0.5 * np.sum((da + db) * hcore)
        expected_energy += 0.5 * np.sum(da * result.fock_alpha + db * result.fock_beta)
        self.assertAlmostEqual(result.energy, expected_energy)

    def test_converged_closed_and_open_shell_spectrum_is_unshifted(self):
        for open_shell in (False, True):
            with self.subTest(open_shell=open_shell):
                shifted = _matrix_run(open_shell=open_shell)
                plain = _matrix_run(open_shell=open_shell, shift=0.0)
                result = shifted[0]
                self.assertTrue(result.converged)
                self.assert_consistent_state(*shifted)
                np.testing.assert_allclose(result.mo_energies, plain[0].mo_energies, atol=1e-12)
                np.testing.assert_allclose(result.fock @ result.mo_coeffs,
                                           result.overlap @ result.mo_coeffs * result.mo_energies,
                                           atol=1e-12)
                if open_shell:
                    # The alpha-energy minimum is the third effective-Fock MO.
                    np.testing.assert_array_equal(result.mo_occupations, [2, 0, 1, 0])

    def test_rank_reduced_space_retains_density_and_unshifted_spectrum(self):
        case = _matrix_run(rank_reduced=True)
        self.assertTrue(case[0].converged)
        self.assertEqual(case[0].mo_coeffs.shape, (3, 2))
        self.assert_consistent_state(*case)
        np.testing.assert_allclose(case[0].mo_energies, [-0.6, 0.7], atol=1e-12)

    def test_exhausted_exit_keeps_physical_fock_and_nonzero_residual(self):
        case = _matrix_run(max_iter=1, nonlinear=True)
        result = case[0]
        self.assertFalse(result.converged)
        self.assertEqual(result.n_iter, 1)
        self.assert_consistent_state(*case)
        residual = result.fock @ result.mo_coeffs - result.overlap @ result.mo_coeffs * result.mo_energies
        self.assertGreater(np.linalg.norm(residual), 1e-3)
        self.assertEqual(len(result.scf_trace), 1)

    def test_block_rotations_preserve_selected_spin_densities_and_full_fock(self):
        ns = {"np": np, "solve_dense_fock": _fixture_dense_fock}
        _source_functions("python/vibeqc/periodic_rohf_ewald.py", ns,
                          {"_canonicalize_final_orbitals"})
        rng = np.random.default_rng(487)
        coeffs, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        occupations = np.array([0.0, 2.0, 1.0, 2.0, 0.0])
        fock = rng.normal(size=(5, 5))
        fock = 0.5 * (fock + fock.T)
        original_c, original_f, original_occ = coeffs.copy(), fock.copy(), occupations.copy()
        c, energies, occ = ns["_canonicalize_final_orbitals"](coeffs, occupations, fock)
        for selector in (lambda x: x > 0, lambda x: x == 2):
            np.testing.assert_allclose((c * selector(occ)) @ c.T,
                                       (coeffs * selector(occupations)) @ coeffs.T,
                                       atol=1e-12)
        np.testing.assert_array_equal(coeffs, original_c)
        np.testing.assert_array_equal(fock, original_f)
        np.testing.assert_array_equal(occupations, original_occ)
        projected = c.T @ fock @ c
        np.testing.assert_allclose(np.diag(projected), energies, atol=1e-12)
        for value in (0, 1, 2):
            indices = np.flatnonzero(occ == value)
            block = projected[np.ix_(indices, indices)]
            np.testing.assert_allclose(block, np.diag(np.diag(block)), atol=1e-12)
        self.assertGreater(np.linalg.norm(fock @ c - c * energies), 0.1)

    def test_zero_iterations_returns_explicitly_unconverged_consistent_state(self):
        case = _matrix_run(max_iter=0, nonlinear=True)
        self.assertFalse(case[0].converged)
        self.assertEqual(case[0].n_iter, 0)
        self.assertEqual(case[0].scf_trace, [])
        self.assert_consistent_state(*case)


if __name__ == "__main__":
    unittest.main(verbosity=2)
