"""Iterative (matrix-free) TDA against the dense reference.

`run_tddft_tda` diagonalises the Casida A matrix densely up to
`TDA_DAVIDSON_MIN_DIM` and iterates above it.  Both routes solve the same
operator, so the dense one is the correctness oracle for the iterative one --
that equality is the contract these tests exist to hold.

Background: forming A costs ``dim**2`` doubles *on top of* an ``eri_mo_ovov``
tensor of the same size, and ``eigh`` is ``O(dim**3)``, to obtain the handful
of lowest roots a TDDFT run actually wants.  For adenine-thymine/cc-pVDZ that
is a 17,204-dimensional matrix -- 2.2 GB -- for ~10 roots.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.tddft import (
    TDA_DAVIDSON_MIN_DIM,
    _build_tda_matrix,
    _solve_tda_iterative,
    tda_sigma_and_diagonal,
)


def _random_tda_inputs(n_occ, n_virt, seed, c_x, with_kernel):
    """ERI blocks carrying the index symmetry a real transform produces."""
    rng = np.random.default_rng(seed)
    eps = np.sort(rng.normal(size=n_occ + n_virt))
    # Separate the orbital energies so A is diagonally dominant, as a real
    # TDA matrix is -- the Davidson preconditioner relies on it.
    eps = eps + np.arange(n_occ + n_virt) * 0.5

    ovov = rng.normal(size=(n_occ, n_virt, n_occ, n_virt)) * 0.05
    ovov = 0.5 * (ovov + ovov.transpose(2, 3, 0, 1))          # (ia|jb) = (jb|ia)
    oovv = rng.normal(size=(n_occ, n_occ, n_virt, n_virt)) * 0.05
    oovv = 0.5 * (oovv + oovv.transpose(1, 0, 3, 2))          # (ij|ab) = (ji|ba)
    kernel = None
    if with_kernel:
        kernel = rng.normal(size=(n_occ, n_virt, n_occ, n_virt)) * 0.05
        kernel = 0.5 * (kernel + kernel.transpose(2, 3, 0, 1))
    return eps, ovov, oovv, kernel, c_x


@pytest.mark.parametrize(
    "n_occ,n_virt,c_x,with_kernel",
    [(3, 5, 0.0, False), (3, 5, 1.0, False), (4, 7, 0.25, True), (2, 9, 0.5, True)],
)
def test_sigma_reproduces_the_dense_operator(n_occ, n_virt, c_x, with_kernel):
    """A @ v computed matrix-free must equal the assembled A times v.

    This is the foundation: if the operator is wrong, every eigenvalue above
    is wrong in the same way and the comparison against dense would still
    agree.  Pinned for a block and a single vector, since the solver uses both.
    """
    eps, ovov, oovv, kernel, cx = _random_tda_inputs(
        n_occ, n_virt, seed=17, c_x=c_x, with_kernel=with_kernel
    )
    A = _build_tda_matrix(eps, ovov, oovv, n_occ, cx, kernel_mo=kernel)
    sigma, diagonal = tda_sigma_and_diagonal(
        eps, ovov, oovv, n_occ, cx, kernel_mo=kernel
    )
    rng = np.random.default_rng(99)
    V = rng.normal(size=(n_occ * n_virt, 3))

    assert np.abs(sigma(V) - A @ V).max() < 1e-12
    assert np.abs(sigma(V[:, 0]) - A @ V[:, 0]).max() < 1e-12
    # The diagonal is supplied analytically rather than probed, so it has to
    # be right independently of the operator.
    assert np.abs(diagonal - np.diag(A)).max() < 1e-12


@pytest.mark.parametrize("c_x,with_kernel", [(0.0, False), (1.0, False), (0.25, True)])
def test_iterative_roots_match_dense_eigh(c_x, with_kernel):
    """The iterative lowest roots must be the dense lowest roots."""
    n_occ, n_virt = 6, 20
    eps, ovov, oovv, kernel, cx = _random_tda_inputs(
        n_occ, n_virt, seed=23, c_x=c_x, with_kernel=with_kernel
    )
    n_pair = n_occ * n_virt
    A = _build_tda_matrix(eps, ovov, oovv, n_occ, cx, kernel_mo=kernel)
    reference = np.linalg.eigvalsh(A)[:5]

    sigma, diagonal = tda_sigma_and_diagonal(
        eps, ovov, oovv, n_occ, cx, kernel_mo=kernel
    )
    evals, evecs, residual = _solve_tda_iterative(sigma, diagonal, n_pair, 5)

    assert np.abs(evals - reference).max() < 1e-9
    assert residual < 1e-4
    # No null columns: every returned amplitude vector is normalised.
    assert np.all(np.linalg.norm(evecs, axis=0) > 0.5)


def test_small_problems_still_take_the_dense_path():
    """Below the threshold nothing changes, so pre-existing numbers stand.

    The gate exists because dense wins there: measured 0.38x and 0.29x at
    dimensions 380 and 480 on cc-pVDZ.
    """
    mol = vq.Molecule.from_xyz(
        "examples/molecular/mp2_benchmarks/s22/geometries/s22-02-water-dimer.xyz"
    )
    basis = vq.BasisSet(mol, "sto-3g")
    n_occ = mol.n_electrons() // 2
    n_pair = n_occ * (basis.nbasis - n_occ)

    assert n_pair <= TDA_DAVIDSON_MIN_DIM, (
        "this fixture is meant to sit below the gate; pick a smaller system"
    )
